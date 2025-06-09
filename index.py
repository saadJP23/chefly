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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image as PILImage
import io
from flask_mail import Mail, Message
from dotenv import load_dotenv
from wtforms import StringField, SubmitField, TextAreaField
import json
import http.client as http_client
from openai import OpenAI

load_dotenv()



# base_dir = os.path.dirname(os.path.abspath(__file__))
# train_dir = os.path.join(base_dir, 'dataset_split', 'training_set')
# class_labels = sorted(os.listdir(train_dir))


# with open('class_labels.json', 'w') as f:
#     json.dump(class_labels, f)



app = Flask(
    __name__
)

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
# CLOUDINARY_CLOUD_NAME=os.getenv('CLOUDINARY_CLOUD_NAME'),
# CLOUDINARY_API_KEY=os.getenv('CLOUDINARY_API_KEY'),
# CLOUDINARY_API_SECRET=os.getenv('CLOUDINARY_API_SECRET'),

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    __tablename__ = 'users'  # Explicitly set table name
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)  # Increased length for hash
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
    image_url = db.Column(db.String(255), nullable=False)
    calories = db.Column(db.Integer, nullable=False)

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
# Image Model
class Image(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(255), nullable=False)

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


@app.route('/generate', methods=['GET', 'POST'])
def generate_recipe():
    pass
    # recipe = {"title": "", "ingredients": [], "instructions": []}

    # if request.method == 'POST':
    #     prompt = request.form.get('prompt')
    #     if prompt:
    #         from transformers import pipeline
    #         generator = pipeline("text-generation", model="gpt2-recipes", tokenizer="gpt2-recipes")

    #         output = generator(
    #             prompt,
    #             max_new_tokens=256,
    #             do_sample=True,
    #             temperature=0.9,
    #             truncation=True,
    #             pad_token_id=50256
    #         )

    #         text = output[0]['generated_text']
    #         print("== RAW MODEL OUTPUT ==")
    #         print(text)

    #         # Stop at first <end>
    #         text = text.split("<end>")[0] if "<end>" in text else text

    #         lines = text.splitlines()
    #         current = None
    #         for idx, line in enumerate(lines):
    #             line = line.strip()
    #             if not line:
    #                 continue
    #             if line.lower().startswith("title:"):
    #                 recipe["title"] = line.replace("Title:", "").strip()
    #             elif idx == 0 and not line.lower().startswith("title:"):
    #                 # Use first line as title if Title: missing
    #                 recipe["title"] = line
    #             elif line.lower().startswith("ingredients:"):
    #                 current = "ingredients"
    #             elif line.lower().startswith("instructions:"):
    #                 current = "instructions"
    #             elif current == "ingredients" and ("-" in line or "•" in line):
    #                 recipe["ingredients"].append(line.lstrip("-• ").strip())
    #             elif current == "instructions" and (
    #                 line[0].isdigit() or line.lower().startswith("step") or line.startswith("-")
    #             ):
    #                 recipe["instructions"].append(line.strip())

    # return render_template('generate.html', recipe=recipe)

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
            
            
            # print(f"GPT response: {content}")
            conn = http_client.HTTPConnection('google.serper.dev')
            payload = json.dumps({
                "q": query})
            headers = {
                'X-API-KEY': os.getenv('SERPER_API_KEY'),
                'Content-Type': 'application/json'
            }
            
            # print(f"Serper API request payload: {payload}")
            conn.request("POST", "/search", payload, headers)
            
            
            # print("Passed Serper API request")
            res = conn.getresponse()
            
            
            # print(f"Serper API response:")
            raw_bytes = res.read()
            # print("Raw bytes:", raw_bytes)
            data = raw_bytes.decode('utf-8')
            # print("Decoded data:", repr(data))

            
            # print("utf-8 printed with: ", data)
            # data_json = json.loads(data)
            # print(f"Serper API response:")
            # image_url = data_json.get("knowledgeGraph", {}).get("imageUrl")
            
            recipe_data = json.loads(content)
            ingredients = recipe_data.get("ingredients", [])
            instructions = recipe_data.get("instructions", "")
            print(f"Calories:")
            
            calories = recipe_data.get("calory", "")
            
            print(f"Calling GPT for url")
        except json.JSONDecodeError:
            ingredients = []
            instructions = content
            
        

        # print(f"initializing gpt_recipe")

        # response = client.images.generate(
        #     prompt=f"Recipe image for {query}",
        #     n=1,
        #     size="1024x1024"
        # )
        
        # image_url = response.data[0].url


        new = Recipe(
            name=query,
            ingredients=json.dumps(ingredients),
            instructions=instructions,
            calories=calories,
            # image_url=image_url,
        )
        db.session.merge(new)
        db.session.commit()
        
        id = None
        saved_recipe = Recipe.query.filter(Recipe.name.ilike(new.name)).first()
        
        
        print(f"Checking for existing recipe with name: {query}")
        print(f"saved_recipe: {saved_recipe}")
        
        
        if saved_recipe:
            id = saved_recipe.id
            print(f"Found existing recipe with ID: {id}")
        else:
            print("No existing recipe found, using None for ID")
        
        
        gpt_recipe = {
            'id': id,
            'name': query,
            'ingredients': ingredients,
            'instructions': instructions,
            'calories': 'null',
            # 'image_url': image_url,        
            }
        # print(f"adding gpt_recipe to results")
        results.append(gpt_recipe)
        
        # Save GPT recipe to database
        # print(f"initializing new Recipe object")
        
        
        # print(f"merging new Recipe object into database")

        
        
        # print(f"calling search_results.html")
        return render_template('search_results.html', results=results)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        flash(f"Error searching recipes: {str(e)}")
        return redirect(url_for('home'))

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

def load_chef_model():
    """Load the MobileNetV2 model for dish prediction"""
    try:
        # Get absolute path to model file
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chefly_mobilenetv2.h5')
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        # Load the model directly
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_dish(image_data, model):
    """Predict the dish from the image data"""
    try:
        # Convert image data to PIL Image
        image = PILImage.open(io.BytesIO(image_data))
        
        # Resize image to match model's expected input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and preprocess
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Same rescaling as training
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        
        # Map class index to dish name
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(BASE_DIR, 'class_labels.json')

        with open(json_path, 'r') as f:
           class_labels = json.load(f)

        
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        if (confidence < 0.5):
            print("Low confidence prediction, returning None")
            return None, 0.0
        
        else:
            predicted_dish = class_labels[predicted_class]
            print(f"Predicted dish: {predicted_dish} with confidence: {confidence:.2f}%")
        
        return predicted_dish, confidence
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, 0.0

@app.route('/upload', methods=['GET'])
def upload_form():
    """Render the upload form"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload a PNG, JPG, JPEG, or GIF file.'}), 400
        
        # Read the image file
        image_data = file.read()
        
        # Load the model
        print("Loading prediction model...")
        model = load_chef_model()
        if model is None:
            error_msg = "Failed to load prediction model. Please check the console for detailed error messages."
            print(error_msg)
            return jsonify({'error': error_msg}), 500
        
        # Make prediction
        print("Making prediction...")
        predicted_dish, confidence = predict_dish(image_data, model)
        
        if predicted_dish is None:
            error_msg = "Failed to predict dish due to low confidence. Please try a different image."
            print(error_msg)
            return jsonify({'error': error_msg}), 500
        
        # Upload image to Cloudinary
        upload_result = cloudinary.uploader.upload(io.BytesIO(image_data), resource_type="image")
        url = upload_result['secure_url']
        
        print(f"Prediction successful: {predicted_dish} ({confidence:.2f})")
        return jsonify({
            'url': url,
            'predicted_dish': predicted_dish,
            'confidence': float(confidence)
        }), 201
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

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

# Add this after your app initialization


@app.template_filter('from_json')
def from_json(value):
    """Convert JSON string to Python object"""
    try:
        return json.loads(value)
    except:
        return value

if __name__ == '__main__':
    # Initialize the database
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8080)

