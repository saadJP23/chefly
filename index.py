from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import requests
import os
import json
import cloudinary
import cloudinary.uploader
import cloudinary.api
import re
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input # Changed to efficientnet
from PIL import Image as PILImage
import io
from flask_mail import Mail, Message
from dotenv import load_dotenv
from wtforms import StringField, SubmitField, TextAreaField
import http.client as http_client
from openai import OpenAI
from tensorflow.keras.preprocessing.image import load_img, img_to_array # Redundant, but keeping for safety if used elsewhere
from bing_image_downloader import downloader
import glob



load_dotenv()

app = Flask(__name__)

app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL') == 'True'
app.config['RESET_SALT'] = os.getenv('RESET_SALT')
mail = Mail(app)

# Cloudinary Configuration
cloudinary.config(
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key = os.getenv('CLOUDINARY_API_KEY'),
    api_secret = os.getenv('CLOUDINARY_API_SECRET')
)

# Image Upload Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
SPOON_API_KEY=os.getenv('SPOON_API_KEY')
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Recipe(db.Model):
    __tablename__ = 'recipe'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    ingredients = db.Column(db.String(255), nullable=False)
    instructions = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(255), nullable=True)
    calories = db.Column(db.Integer, nullable=True)
    
class Favorites(db.Model):
    __tablename__ = 'Favorites'
    UserId = db.Column(db.Integer, primary_key=True)
    RecipeId = db.Column(db.Integer, primary_key=True)

# Forms
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    subject = StringField('Subject', validators=[DataRequired(), Length(max=100)])
    message = TextAreaField('Message', validators=[DataRequired()])
    submit = SubmitField('Send Message')

# Image Model (was in your provided index.py, but not strictly used with new upload flow)
class Image(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(255), nullable=False)

class TrainedDish(db.Model):
    __tablename__ = 'Trained_dishes'
    dish_id = db.Column(db.Integer, primary_key=True)
    dish = db.Column(db.String(100), nullable=False)


# --- GLOBAL MODEL AND LABEL LOADING ---
# Global variables for the model and class labels to avoid reloading on every request
chefly_model = None
class_labels_list = None

def load_chef_model_and_labels():
    """Load the EfficientNetB0 model and class labels once."""
    global chefly_model, class_labels_list
    if chefly_model is None:
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chefly_EfficientNetB0.h5')
            print(f"Loading model from: {model_path}")

            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return False, "Model file not found."

            chefly_model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully")

            # Load class labels
            labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_labels.json')
            if not os.path.exists(labels_path):
                print(f"Error: class_labels.json not found at {labels_path}")
                return False, "Class labels file not found."
            with open(labels_path, "r") as f:
                class_labels_list = json.load(f)
            print("Class labels loaded successfully")

            return True, None
        except Exception as e:
            print(f"Error loading model or labels: {str(e)}")
            chefly_model = None
            class_labels_list = None
            return False, f"Failed to load model or labels: {str(e)}"
    return True, None

# Call this once when the app starts
with app.app_context():
    success, error_msg = load_chef_model_and_labels()
    if not success:
        print(f"FATAL ERROR: Could not load machine learning model: {error_msg}")
        # Depending on your deployment, you might want to exit or log this more seriously

# --- END GLOBAL MODEL AND LABEL LOADING ---

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if not user:
            flash('Email not found. Please sign up first.', 'danger')
            return redirect(url_for('login'))

        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('home'))
        flash('Login unsuccessful. Please check your email and password.', 'danger')
    return render_template('login.html', form=form)



@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({'message': 'Email not found'}), 404

    if user and user.check_password(password):
        return jsonify({'message': 'Login successful'}), 200

    return jsonify({'message': 'Invalid password'}), 401
@app.route('/generate', methods=['GET', 'POST'])
def generate_recipe():
    pass
 
from flask import request, jsonify

@app.route('/api/register', methods=['POST'])
def register_api():
    data = request.get_json()

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Validation
    if not username or not email or not password:
        return jsonify({"success": False, "message": "All fields are required."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email already registered."}), 409

    if User.query.filter_by(username=username).first():
        return jsonify({"success": False, "message": "Username already taken."}), 409

    # Register new user
    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"success": True, "message": "Registration successful."}), 201
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if email already exists
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered. Please use a different email or login.', 'danger')
            return render_template('signup.html', form=form)
        # Check if username already exists
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already taken. Please choose a different username.', 'danger')
            return render_template('signup.html', form=form)

        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/search')
def search():
    try:
        query = request.args.get('q', '')
        results = []

        # First, search in local database
        dish = Recipe.query.filter(Recipe.name.ilike(f'%{query}%')).all()
        if dish:
            return render_template('search_results.html', query=query, results=dish)

        # If not found locally, try Spoonacular API
        spoon_url = 'https://api.spoonacular.com/recipes/complexSearch'
        params = {
            'query': query,
            'number': 1,
            'apiKey': SPOON_API_KEY
        }
        spoon_res = requests.get(spoon_url, params=params)
        data = spoon_res.json().get('results', [])

        if data:
            r = data[0]
            info_resp = requests.get(
                f"https://api.spoonacular.com/recipes/{r['id']}/information",
                params={'apiKey': SPOON_API_KEY, 'includeNutrition': 'true'}
            )
            info = info_resp.json()

            def clean_html(text):
                """Remove HTML tags from text"""
                if not text:
                    return ""
                # Remove HTML tags
                clean = re.compile('<.*?>')
                text = re.sub(clean, '', text)
                # Replace multiple newlines with single newline
                text = re.sub(r'\n\s*\n', '\n', text)
                # Replace &nbsp; with space
                text = text.replace('&nbsp;', ' ')
                return text.strip()

            recipe = {
                'id': info['id'],
                'name': info['title'],
                'ingredients': [ing['original'] for ing in info.get('extendedIngredients', [])],
                'instructions': clean_html(info.get('instructions') or 'No instructions provided.'),
                'calories': next(
                    (n['amount'] for n in info.get('nutrition', {}).get('nutrients', []) if n['name'] == 'Calories'),
                    None
                ),
                'image_url': info.get('image'),
            }
            results.append(recipe)

            # Save to local database
            new = Recipe(
                id=r['id'],
                name=r['title'],
                ingredients=json.dumps(recipe['ingredients']),
                instructions=recipe['instructions'],
                calories=str(recipe['calories']),
                image_url=recipe['image_url'],
            )
            db.session.merge(new)
            db.session.commit()

            return render_template('search_results.html', results=results)

        # If not found in Spoonacular, try GPT
        try:
            print("Calling GPT-turbo for recipe generation...")
            prompt = f"""
            You are a cooking assistant. Please output a JSON object exactly in this format, with no extra text:
            {{
                "ingredients": [
                    "ingredient 1",
                    "ingredient 2",
                    …
                ],
                "instructions": "Step-by-step cooking instructions.",
                "calory": "Approximate total calories for a standard serving of this recipe. Please provide a number only, not a description.",
            }}
            Now, give me the ingredients and instructions for how to make '{query}'.
            """

            

            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content
            flash("✅ Generated by GPT-turbo", "warning")

            conn = http_client.HTTPConnection('google.serper.dev')
            payload = json.dumps({
                "q": query})
            headers = {
                'X-API-KEY': os.getenv('SERPER_API_KEY'),
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/search", payload, headers)

            res = conn.getresponse()

            raw_bytes = res.read()
            data = raw_bytes.decode('utf-8')

            recipe_data = json.loads(content)
            ingredients = recipe_data.get("ingredients", [])
            instructions = recipe_data.get("instructions", "")
            calories = recipe_data.get("calory", None) # Changed to None for consistency

            response_image_gen = client.images.generate(
                prompt=f"Recipe image for {query}",
                n=1,
                size="1024x1024"
            )

            image_url = response_image_gen.data[0].url

            downloader.download(query, limit=1, output_dir='bing_images', adult_filter_off=True, force_replace=False, timeout=60)

            image_path_list = glob.glob(f'bing_images/{query}/*')
            image_path = image_path_list[0] if image_path_list else None
            

            if image_path:
                upload_result = cloudinary.uploader.upload(image_path, folder="recipes", public_id=f"{query}_image")
                image_url = upload_result['secure_url']
            else:
                image_url = None

        except json.JSONDecodeError:
            ingredients = []
            instructions = content
            image_url = None # Set to None if JSON parsing fails
            calories = None

        new = Recipe(
            name=query,
            ingredients=json.dumps(ingredients),
            instructions=instructions,
            calories=calories,
            image_url=image_url,
        )
        db.session.add(new) # Changed from merge to add, assuming new recipe
        db.session.commit()

        id = new.id # Get ID after commit

        gpt_recipe = {
            'id': id,
            'name': query,
            'ingredients': ingredients,
            'instructions': instructions,
            'calories': calories,
            'image_url': image_url,
            }
        results.append(gpt_recipe)

        os.remove(image_path)
        os.rmdir(f'bing_images/{query}')

        return render_template('search_results.html', results=results)

    except Exception as e:
        print(f"❌ Error: {e}")
        flash(f"Error searching recipes: {str(e)}")
        return redirect(url_for('home'))


@app.route('/api/favorites', methods=["GET"])
def api_favorites():
    user_id = request.args.get('user_id') 
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    dishes = db.session.execute("""
        SELECT r.* FROM favorites f
        INNER JOIN recipe r ON f.RecipeId = r.id
        WHERE f.UserId = :user_id
    """, {'user_id': user_id}).fetchall()


    dish_list = [{
        'id': row.id,
        'name': row.name,
        'ingredients': row.ingredients,
        'instructions': row.instructions,
        'image_url': row.image_url,
        'calories': row.calories
    } for row in dishes]

    return jsonify(dish_list)




    



@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        
        data = request.get_json()
        query = data.get('query', '')

        if not query:
            return jsonify({"error": "Query cannot be empty."}), 400

        results = []

        # Search in local DB
        dish = Recipe.query.filter(Recipe.name.ilike(f'%{query}%')).all()
        if dish:
            return jsonify([{
                'id': d.id,
                'name': d.name,
                'ingredients': json.loads(d.ingredients),
                'instructions': d.instructions,
                'calories': d.calories,
                'image_url': d.image_url
            } for d in dish]), 200

        # If not found, try Spoonacular
        spoon_url = 'https://api.spoonacular.com/recipes/complexSearch'
        spoon_params = {
            'query': query,
            'number': 1,
            'apiKey': SPOON_API_KEY
        }
        spoon_res = requests.get(spoon_url, params=spoon_params)
        spoon_data = spoon_res.json().get('results', [])

        if spoon_data:
            r = spoon_data[0]
            info = requests.get(
                f"https://api.spoonacular.com/recipes/{r['id']}/information",
                params={'apiKey': SPOON_API_KEY, 'includeNutrition': 'true'}
            ).json()

            def clean_html(text):
                clean = re.compile('<.*?>')
                return re.sub(clean, '', text or '').replace('&nbsp;', ' ').strip()

            recipe = {
                'id': info['id'],
                'name': info['title'],
                'ingredients': [i['original'] for i in info.get('extendedIngredients', [])],
                'instructions': clean_html(info.get('instructions')),
                'calories': next((n['amount'] for n in info.get('nutrition', {}).get('nutrients', []) if n['name'] == 'Calories'), None),
                'image_url': info.get('image'),
            }

            # Save to DB
            db.session.merge(Recipe(
                id=r['id'],
                name=recipe['name'],
                ingredients=json.dumps(recipe['ingredients']),
                instructions=recipe['instructions'],
                calories=str(recipe['calories']),
                image_url=recipe['image_url'],
            ))
            db.session.commit()

            return jsonify([recipe]), 201

        try:
            print("Calling GPT-turbo for recipe generation...")
            prompt = f"""
            You are a cooking assistant. Please output a JSON object exactly in this format, with no extra text:
            {{
                "ingredients": [
                    "ingredient 1",
                    "ingredient 2",
                    …
                ],
                "instructions": "Step-by-step cooking instructions.",
                "calory": "Approximate total calories for a standard serving of this recipe. Please provide a number only, not a description.",
            }}
            Now, give me the ingredients and instructions for how to make '{query}'.
            """

            

            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content
            flash("✅ Generated by GPT-turbo", "warning")

            conn = http_client.HTTPConnection('google.serper.dev')
            payload = json.dumps({
                "q": query})
            headers = {
                'X-API-KEY': os.getenv('SERPER_API_KEY'),
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/search", payload, headers)

            res = conn.getresponse()

            raw_bytes = res.read()
            data = raw_bytes.decode('utf-8')

            recipe_data = json.loads(content)
            ingredients = recipe_data.get("ingredients", [])
            instructions = recipe_data.get("instructions", "")
            calories = recipe_data.get("calory", None) # Changed to None for consistency

            response_image_gen = client.images.generate(
                prompt=f"Recipe image for {query}",
                n=1,
                size="1024x1024"
            )

            image_url = response_image_gen.data[0].url

            downloader.download(query, limit=1, output_dir='bing_images', adult_filter_off=True, force_replace=False, timeout=60)

            image_path_list = glob.glob(f'bing_images/{query}/*')
            image_path = image_path_list[0] if image_path_list else None
            

            if image_path:
                upload_result = cloudinary.uploader.upload(image_path, folder="recipes", public_id=f"{query}_image")
                image_url = upload_result['secure_url']
            else:
                image_url = None

        except json.JSONDecodeError:
            ingredients = []
            instructions = content
            image_url = None # Set to None if JSON parsing fails
            calories = None

        new = Recipe(
            name=query,
            ingredients=json.dumps(ingredients),
            instructions=instructions,
            calories=calories,
            image_url=image_url,
        )
        db.session.add(new) # Changed from merge to add, assuming new recipe
        db.session.commit()

        id = new.id # Get ID after commit

        gpt_recipe = {
            'id': id,
            'name': query,
            'ingredients': ingredients,
            'instructions': instructions,
            'calories': calories,
            'image_url': image_url,
            }
        results.append(gpt_recipe)

        os.remove(image_path)
        os.rmdir(f'bing_images/{query}')

        return jsonify([gpt_recipe]), 201

    except Exception as e:
        print(f"Error in /api/search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Render the contact page"""

    form = ContactForm()
    msg_sent = False

    if(form.validate_on_submit()):
        name = form.name.data
        email = form.email.data
        message = form.message.data

        if not name or not email or not message:
            flash("All fields are required", "danger")
            return redirect(url_for('contact'))

        msg = Message("New Contact Form Submission",
                      sender=email,
                      recipients =[app.config['MAIL_USERNAME']])

        msg.body=f"""
        New message from {name}:

        Email: {email}
        Message: {message}"""

        try:
            mail.send(msg)
            flash("Your message has been sent successfully!", "success")
            msg_sent = True

        except Exception as e:
            print(f"Error sending email: {e}")
            flash("There was an error sending your message. Please try again later.", "danger")

        return redirect(url_for('contact'))

    return render_template('contact.html', form=form, msg_sent=msg_sent)

@app.route('/famous-dishes')
def famous_dishes():
    try:
        # Get total count of recipes
        total_recipes = Recipe.query.count()

        if total_recipes == 0:
            flash("No recipes available", "info")
            return redirect(url_for('home'))

        # Generate 5 random indices
        random_indices = random.sample(range(total_recipes), min(5, total_recipes))

        # Fetch random recipes
        random_dishes = []
        for idx in random_indices:
            dish = Recipe.query.offset(idx).first()
            if dish:
                random_dishes.append(dish)

        return render_template('famous_dishes.html', dishes=random_dishes)

    except Exception as e:
        print(f"Error fetching famous dishes: {e}")
        flash("Error loading famous dishes", "error")


        return redirect(url_for('home'))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- REVISED PREDICT_DISH FUNCTION ---
def predict_dish(image_data):
    """
    Predicts the dish from the given image data using the loaded model.
    Args:
        image_data: Bytes of the image file.
    Returns:
        tuple: (predicted_label, confidence, error_message)
    """
    global chefly_model, class_labels_list

    # Ensure model and labels are loaded
    if chefly_model is None or class_labels_list is None:
        success, error = load_chef_model_and_labels()
        if not success:
            return None, None, error

    print("Making prediction...")
    try:
        # Load and preprocess image
        image = PILImage.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict using the loaded model
        predictions = chefly_model.predict(img_array)[0]

        # Validate output shape matches label count
        if len(predictions) != len(class_labels_list):
            return None, None, "Mismatch between model output and class_labels.json. Model output shape might be incorrect."

        # Get top prediction and top-5 for logging
        top_5 = np.argsort(predictions)[::-1][:5]
        top_label = class_labels_list[top_5[0]]
        confidence = predictions[top_5[0]]

        print(f"\n✅ Predicted: {top_label} ({confidence*100:.2f}% confidence)\n")
        print("🔝 Top 5 Predictions:")
        for i in top_5:
            print(f"{class_labels_list[i]}: {predictions[i] * 100:.2f}%")

        return top_label, confidence, None # Return prediction, confidence, and no error

    except Exception as e:
        print("Error making prediction:", e)
        return None, None, str(e) # Return None for prediction and confidence, and error message
# --- END REVISED PREDICT_DISH FUNCTION ---


@app.route('/api/trained', methods=["GET"])
def api_trained_dishes():
    query = TrainedDish.query.all()

    dishes = []
    for dish in query:
        dish_data = {
            'id': dish.dish_id,
            "dish": dish.dish
        }
        dishes.append(dish_data)

    return jsonify(dishes), 200


@app.route('/trained_dishes', methods=["GET"])
def trained_dishes():

    query = TrainedDish.query.all()
    dishes = []
    for dish in query:
        dish_data = {
            'id': dish.dish_id,
            "dish": dish.dish
        }
        dishes.append(dish_data)


    return render_template('trained_dishes.html', dishes = dishes)

@app.route('/upload', methods=['GET'])
def upload_form():
    """Render the upload form"""
    return render_template('upload.html')

# --- REVISED UPLOAD_FILE FUNCTION ---
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # 1. Check for file in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 2. Validate file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload a PNG, JPG, JPEG, or GIF file.'}), 400

        # 3. Read image data
        image_data = file.read()

        # 4. Perform prediction
        predicted_dish, confidence, prediction_error = predict_dish(image_data)

        if prediction_error:
            return jsonify({'error': f"Prediction failed: {prediction_error}"}), 500

        # 5. Upload image to Cloudinary (only if prediction was successful)
        try:
            upload_result = cloudinary.uploader.upload(io.BytesIO(image_data), resource_type="image")
            image_url = upload_result['secure_url']
        except Exception as e:
            return jsonify({'error': f"Failed to upload image to Cloudinary: {str(e)}"}), 500

        # 6. Return successful response
        return jsonify({
            'url': image_url,
            'predicted_dish': predicted_dish,
            'confidence': float(confidence)
        }), 200

    except Exception as e:
        error_msg = f"An unexpected error occurred during upload: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500
# --- END REVISED UPLOAD_FILE FUNCTION ---


@app.route('/confirm-prediction', methods=['POST'])
def confirm_prediction():
    """Handle prediction confirmation"""
    try:
        dish_name = request.form.get('dish_name')
        is_confirmed = request.form.get('confirmed') == 'yes'

        if is_confirmed:
            # Find the recipe in database
            recipe = Recipe.query.filter(Recipe.name.ilike(f'%{dish_name}%')).first()
            if recipe:
                return redirect(url_for('view_recipe', recipe_id=recipe.id))
            else:
                flash('Recipe not found in database', 'error')
                return redirect(url_for('upload_form'))
        else:
            # Show manual input form
            return render_template('image_search_result.html',
                                predicted_name=None,
                                confidence=0,
                                image_url=request.form.get('image_url'),
                                show_manual_input=True)

    except Exception as e:
        print(f"Error in confirm_prediction: {e}")
        flash('Error processing your request', 'error')
        return redirect(url_for('upload_form'))

@app.route('/recipe/<int:recipe_id>')
def view_recipe(recipe_id):
    """Display detailed recipe information"""
    try:
        # Try to find recipe in database first
        recipe = Recipe.query.get_or_404(recipe_id)
        return render_template('recipe.html', recipe=recipe)

    except Exception as e:
        flash(f"Error loading recipe: {str(e)}", "error")
        return redirect(url_for('home'))
    
from flask import jsonify

@app.route('/api/recipe/<int:recipe_id>', methods=['GET'])
def api_view_recipe(recipe_id):
    try:
        recipe = Recipe.query.get_or_404(recipe_id)
        return jsonify({
            "id": recipe.id,
            "title": recipe.title,
            "ingredients": recipe.ingredients.split(',') if isinstance(recipe.ingredients, str) else recipe.ingredients,
            "instructions": recipe.instructions,
            "calories": recipe.calories,
            "image_url": recipe.image_url
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.template_filter('from_json')
def from_json(value):
    """Convert JSON string to Python object"""
    try:
        return json.loads(value)
    except:
        return value

@app.route('/start_search', methods=['POST'])
def start_search():
    query = request.form.get('q', '')
    return render_template('loading_page.html', query=query)

if __name__ == '__main__':
    # Initialize the database
    with app.app_context():
        db.create_all()
        
    app.run(host='0.0.0.0', port=8080, debug=True)