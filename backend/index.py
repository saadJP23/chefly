from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, EqualTo, Length, ValidationError
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import requests
import os
import json
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
except ModuleNotFoundError:
    cloudinary = None
import re
import random
import functools
import html
import shutil
from urllib.parse import quote
import numpy as np
from PIL import Image as PILImage
import io
from flask_mail import Mail, Message
from flask_cors import CORS
from dotenv import load_dotenv
from wtforms import StringField, SubmitField, TextAreaField
import http.client as http_client
try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None
try:
    from bing_image_downloader import downloader
except ModuleNotFoundError:
    downloader = None
import glob
import secrets
from datetime import datetime, timedelta
from sqlalchemy import func, or_
from flask_cors import CORS

try:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
except ModuleNotFoundError:
    tf = None
    preprocess_input = None
    img_to_array = None



load_dotenv()

app = Flask(__name__)
application = app
CORS(app, origins=[
    "https://cheflys.com",
    "https://www.cheflys.com",
    "https://web-production-9ad87.up.railway.app"
])

cors_origins = os.getenv('CORS_ORIGINS', '*')
CORS(
    app,
    origins=[origin.strip() for origin in cors_origins.split(',')] if cors_origins != '*' else '*',
    supports_credentials=True
)

app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT') or 587)
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL') == 'True'
app.config['RESET_SALT'] = os.getenv('RESET_SALT')
mail = Mail(app)

# Cloudinary Configuration
if cloudinary:
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
client = OpenAI(api_key='OPENAI_API_KEY') if OpenAI else None

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
    avatar_url = db.Column(db.String(255), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    dietary_pref = db.Column(db.String(120), nullable=True)
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
    category = db.Column(db.String(80), nullable=True)
    cuisine = db.Column(db.String(80), nullable=True)
    diet_type = db.Column(db.String(80), nullable=True)
    prep_time_mins = db.Column(db.Integer, nullable=True)
    cook_time_mins = db.Column(db.Integer, nullable=True)
    servings = db.Column(db.Integer, nullable=True)
    protein_g = db.Column(db.Integer, nullable=True)
    carbs_g = db.Column(db.Integer, nullable=True)
    fat_g = db.Column(db.Integer, nullable=True)
    fiber_g = db.Column(db.Integer, nullable=True)
    submitted_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    is_user_submitted = db.Column(db.Boolean, default=False)
    
class Favorites(db.Model):
    __tablename__ = 'Favorites'
    UserId = db.Column(db.Integer, primary_key=True)
    RecipeId = db.Column(db.Integer, primary_key=True)

class Rating(db.Model):
    __tablename__ = 'ratings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    recipe_id = db.Column(db.Integer, db.ForeignKey('recipe.id'), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    user = db.relationship('User', backref='ratings')
    recipe = db.relationship('Recipe', backref='ratings')
    __table_args__ = (db.UniqueConstraint('user_id', 'recipe_id', name='uq_user_recipe_rating'),)

class PasswordResetToken(db.Model):
    __tablename__ = 'password_reset_tokens'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    token = db.Column(db.String(128), unique=True, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    user = db.relationship('User', backref='password_reset_tokens')

# Forms
def email_format(form, field):
    value = (field.data or '').strip()
    if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', value):
        raise ValidationError('Please enter a valid email address.')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), email_format])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), email_format])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), email_format])
    subject = StringField('Subject', validators=[DataRequired(), Length(max=100)])
    message = TextAreaField('Message', validators=[DataRequired()])
    submit = SubmitField('Send Message')

class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), email_format])
    submit = SubmitField('Send Reset Link')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Update Password')

# Image Model (was in your provided index.py, but not strictly used with new upload flow)
class Image(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(255), nullable=False)

class TrainedDish(db.Model):
    __tablename__ = 'Trained_dishes'
    dish_id = db.Column(db.Integer, primary_key=True)
    dish = db.Column(db.String(100), nullable=False)


def _class_labels_json_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_labels.json')


@functools.lru_cache(maxsize=1)
def _cached_class_labels_tuple():
    path = _class_labels_json_path()
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('class_labels.json must contain a JSON array of strings')
    return tuple(data)


def load_class_labels():
    """Trained class names from class_labels.json (same source as the CNN labels)."""
    return list(_cached_class_labels_tuple())


def trained_dish_rows_from_labels():
    """dict rows with keys id, dish — matches trained_dishes.html and /api/trained."""
    return [{'id': idx, 'dish': label} for idx, label in enumerate(load_class_labels(), start=1)]


def placeholder_dish_image(label_display: str) -> str:
    """Small inline SVG so famous-dish cards work without external image hosts."""
    t = html.escape(label_display[:40], quote=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400">'
        f'<rect fill="#e8dcc4" width="100%" height="100%"/>'
        f'<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" '
        f'fill="#4a3728" font-size="20" font-family="sans-serif">{t}</text></svg>'
    )
    return 'data:image/svg+xml,' + quote(svg)


def famous_dish_cards_from_labels(count=5):
    """Card dicts for famous_dishes.html: name, link, image_url, calories, instructions."""
    labels = load_class_labels()
    if not labels:
        return []
    picks = random.sample(labels, k=min(count, len(labels)))
    cards = []
    for label in picks:
        display = label.replace('_', ' ')
        cards.append({
            'name': display,
            'link': url_for('search', q=display),
            'image_url': placeholder_dish_image(display),
            'calories': '—',
            'instructions': f'Explore recipes and ideas for {display}.',
        })
    return cards


def _openai_message_text(resp):
    if not resp or not getattr(resp, 'choices', None):
        return ''
    msg = resp.choices[0].message
    return (getattr(msg, 'content', None) or '').strip()


def _strip_markdown_json_fences(text):
    t = (text or '').strip()
    if not t.startswith('```'):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith('```'):
        lines = lines[1:]
    while lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines).strip()


def _normalize_gpt_recipe_payload(recipe_data):
    """Coerce GPT JSON into safe DB/template types (avoids None passed to string APIs)."""
    if not isinstance(recipe_data, dict):
        return [], '', None
    raw_ing = recipe_data.get('ingredients', [])
    if raw_ing is None or not isinstance(raw_ing, list):
        raw_ing = []
    ingredients = [str(x) for x in raw_ing if x is not None]
    ins = recipe_data.get('instructions', None)
    instructions = '' if ins is None else str(ins)
    cal = recipe_data.get('calory', None)
    if cal is not None and not isinstance(cal, (str, int, float)):
        cal = str(cal)
    return ingredients, instructions, cal


def _cleanup_bing_download_dir(query, image_path=None):
    if image_path and os.path.isfile(image_path):
        try:
            os.remove(image_path)
        except OSError:
            pass
    bing_dir = os.path.join('bing_images', query)
    if os.path.isdir(bing_dir):
        shutil.rmtree(bing_dir, ignore_errors=True)


# --- GLOBAL MODEL AND LABEL LOADING ---
# Global variables for the model and class labels to avoid reloading on every request
chefly_model = None
class_labels_list = None

def load_chef_model_and_labels():
    """Load the EfficientNetB0 model and class labels once."""
    global chefly_model, class_labels_list
    if tf is None:
        return False, "TensorFlow is not installed in this environment."
    if chefly_model is None:
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chefly_mobilenetv2.h5')
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

def _json_ingredients(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (TypeError, json.JSONDecodeError):
        pass
    return [item.strip() for item in str(value).split(',') if item.strip()]

def _int_or_none(value):
    if value in (None, ''):
        return None
    try:
        return int(float(str(value).replace(' kcal', '').replace('calories', '').strip()))
    except (TypeError, ValueError):
        return None

def _avg_rating(recipe_id):
    avg, count = db.session.query(func.avg(Rating.score), func.count(Rating.id)).filter_by(recipe_id=recipe_id).first()
    return round(float(avg or 0), 1), int(count or 0)

def _recipe_card(recipe):
    rating, rating_count = _avg_rating(recipe.id)
    total_time = (recipe.prep_time_mins or 0) + (recipe.cook_time_mins or 0)
    return {
        'id': recipe.id,
        'name': recipe.name,
        'ingredients': _json_ingredients(recipe.ingredients),
        'instructions': recipe.instructions,
        'image_url': recipe.image_url or placeholder_dish_image(recipe.name),
        'calories': recipe.calories,
        'category': recipe.category,
        'cuisine': recipe.cuisine,
        'diet_type': recipe.diet_type,
        'prep_time_mins': recipe.prep_time_mins,
        'cook_time_mins': recipe.cook_time_mins,
        'servings': recipe.servings,
        'protein_g': recipe.protein_g,
        'carbs_g': recipe.carbs_g,
        'fat_g': recipe.fat_g,
        'fiber_g': recipe.fiber_g,
        'rating': rating or 'New',
        'rating_count': rating_count,
        'cooking_time': f'{total_time} min' if total_time else 'Time varies',
        'description': recipe.instructions[:130] + ('...' if len(recipe.instructions or '') > 130 else ''),
        'tags': [tag for tag in [recipe.category, recipe.cuisine, recipe.diet_type] if tag],
        'is_vegetarian': (recipe.diet_type or '').lower() == 'vegetarian'
    }

def _send_reset_email(user, token):
    reset_url = url_for('reset_password', token=token, _external=True)
    msg = Message('Reset your Chefly password',
                  sender=app.config['MAIL_USERNAME'],
                  recipients=[user.email])
    msg.body = f'Hi {user.username},\n\nReset your Chefly password here:\n{reset_url}\n\nThis link expires in 1 hour.'
    mail.send(msg)

def seed_recipe_metadata():
    """Backfill older rows so filters, time badges, and nutrition have usable data."""
    recipes = Recipe.query.all()
    categories = ['Breakfast', 'Main', 'Dessert', 'Snack', 'Dinner']
    cuisines = ['Pakistani', 'Japanese', 'Italian', 'Thai', 'American', 'Global']
    changed = False
    for recipe in recipes:
        name = (recipe.name or '').lower()
        if not recipe.category:
            if any(word in name for word in ['cake', 'halwa', 'rasmalai', 'baklava', 'zarda', 'gelato']):
                recipe.category = 'Dessert'
            elif any(word in name for word in ['breakfast', 'puri', 'croissant']):
                recipe.category = 'Breakfast'
            elif any(word in name for word in ['vada', 'bun', 'fries', 'samosa']):
                recipe.category = 'Snack'
            else:
                recipe.category = categories[recipe.id % len(categories)]
            changed = True
        if not recipe.cuisine:
            if any(word in name for word in ['biryani', 'nihari', 'karahi', 'palao', 'qeema']):
                recipe.cuisine = 'Pakistani'
            elif any(word in name for word in ['sushi', 'ramen', 'takoyaki']):
                recipe.cuisine = 'Japanese'
            elif any(word in name for word in ['pizza', 'pasta']):
                recipe.cuisine = 'Italian'
            else:
                recipe.cuisine = cuisines[recipe.id % len(cuisines)]
            changed = True
        if not recipe.diet_type:
            meat_words = ['chicken', 'beef', 'lamb', 'duck', 'fish', 'salmon', 'kebab']
            recipe.diet_type = 'Regular' if any(word in name for word in meat_words) else 'Vegetarian'
            changed = True
        if recipe.prep_time_mins is None:
            recipe.prep_time_mins = 10 + (recipe.id % 3) * 5
            changed = True
        if recipe.cook_time_mins is None:
            recipe.cook_time_mins = 20 + (recipe.id % 5) * 10
            changed = True
        if recipe.servings is None:
            recipe.servings = 4
            changed = True
        try:
            calories = int(float(recipe.calories or 350))
        except (TypeError, ValueError):
            calories = 350
        if recipe.protein_g is None:
            recipe.protein_g = max(6, int(calories * 0.08))
            recipe.carbs_g = max(12, int(calories * 0.12))
            recipe.fat_g = max(5, int(calories * 0.04))
            recipe.fiber_g = 3 + (recipe.id % 5)
            changed = True
    if changed:
        db.session.commit()

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
    generated = None
    if request.method == 'POST':
        query = request.form.get('prompt', '').strip()
        if not query:
            flash('Tell Chefly what you want to cook first.', 'warning')
            return render_template('generate.html', generated=None)

        prompt = f"""
        Return only JSON for a recipe matching this request: {query}
        Fields: name, ingredients (array), instructions, calories, category, cuisine, diet_type,
        prep_time_mins, cook_time_mins, servings, protein_g, carbs_g, fat_g, fiber_g.
        """
        content = ''
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            content = _strip_markdown_json_fences(_openai_message_text(resp))
            data = json.loads(content)
        except Exception as exc:
            print(f"AI generate fallback used: {exc}")
            data = {
                'name': query.title(),
                'ingredients': ['1 cup pantry staples', 'Fresh herbs', 'Salt and pepper'],
                'instructions': 'Combine the ingredients, cook until tender, season to taste, and serve warm.',
                'calories': 350,
                'category': 'Main',
                'cuisine': 'Global',
                'diet_type': 'Vegetarian',
                'prep_time_mins': 10,
                'cook_time_mins': 25,
                'servings': 4,
                'protein_g': 12,
                'carbs_g': 42,
                'fat_g': 14,
                'fiber_g': 6,
            }

        ingredients = data.get('ingredients') if isinstance(data.get('ingredients'), list) else []
        recipe = Recipe(
            name=str(data.get('name') or query.title()),
            ingredients=json.dumps(ingredients),
            instructions=str(data.get('instructions') or ''),
            calories=_int_or_none(data.get('calories')),
            category=data.get('category') or 'Main',
            cuisine=data.get('cuisine') or 'Global',
            diet_type=data.get('diet_type') or 'Regular',
            prep_time_mins=_int_or_none(data.get('prep_time_mins')),
            cook_time_mins=_int_or_none(data.get('cook_time_mins')),
            servings=_int_or_none(data.get('servings')) or 4,
            protein_g=_int_or_none(data.get('protein_g')),
            carbs_g=_int_or_none(data.get('carbs_g')),
            fat_g=_int_or_none(data.get('fat_g')),
            fiber_g=_int_or_none(data.get('fiber_g')),
            submitted_by=current_user.id if current_user.is_authenticated else None,
        )
        db.session.add(recipe)
        db.session.commit()
        generated = recipe
        flash('Recipe generated and saved.', 'success')
    return render_template('generate.html', generated=generated)
 
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
        category = request.args.get('category', '')
        cuisine = request.args.get('cuisine', '')
        diet_type = request.args.get('diet_type', '')
        max_calories = _int_or_none(request.args.get('max_calories'))
        max_time = _int_or_none(request.args.get('max_time'))
        sort = request.args.get('sort', 'relevance')
        results = []

        # First, search in local database
        dish_query = Recipe.query
        if query:
            dish_query = dish_query.filter(or_(Recipe.name.ilike(f'%{query}%'), Recipe.ingredients.ilike(f'%{query}%')))
        if category:
            dish_query = dish_query.filter(Recipe.category.ilike(category))
        if cuisine:
            dish_query = dish_query.filter(Recipe.cuisine.ilike(cuisine))
        if diet_type:
            dish_query = dish_query.filter(Recipe.diet_type.ilike(diet_type))
        if max_calories is not None:
            dish_query = dish_query.filter(Recipe.calories <= max_calories)
        if max_time is not None:
            dish_query = dish_query.filter((func.coalesce(Recipe.prep_time_mins, 0) + func.coalesce(Recipe.cook_time_mins, 0)) <= max_time)
        if sort == 'newest':
            dish_query = dish_query.order_by(Recipe.id.desc())
        elif sort == 'calories':
            dish_query = dish_query.order_by(Recipe.calories.asc())
        dish = dish_query.limit(60).all()
        if dish:
            return render_template('search_results.html', query=query, results=[_recipe_card(d) for d in dish])

        if category or cuisine or diet_type or max_calories or max_time:
            return render_template('search_results.html', query=query, results=[])

        # If not found locally, try Spoonacular API
        spoon_url = 'https://api.spoonacular.com/recipes/complexSearch'
        params = {
            'query': query,
            'number': 1,
            'apiKey': SPOON_API_KEY
        }
        try:
            spoon_res = requests.get(spoon_url, params=params, timeout=12)
            data = spoon_res.json().get('results', [])
        except requests.RequestException as exc:
            print(f"Spoonacular search skipped: {exc}")
            data = []

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
                'category': 'Main',
                'cuisine': info.get('cuisines', ['Global'])[0] if info.get('cuisines') else 'Global',
                'diet_type': 'Vegetarian' if info.get('vegetarian') else 'Regular',
                'prep_time_mins': info.get('preparationMinutes') or 10,
                'cook_time_mins': info.get('cookingMinutes') or info.get('readyInMinutes') or 30,
                'servings': info.get('servings') or 4,
            }

            # Save to local database
            new = Recipe(
                id=r['id'],
                name=r['title'],
                ingredients=json.dumps(recipe['ingredients']),
                instructions=recipe['instructions'],
                calories=str(recipe['calories']),
                image_url=recipe['image_url'],
                category=recipe['category'],
                cuisine=recipe['cuisine'],
                diet_type=recipe['diet_type'],
                prep_time_mins=recipe['prep_time_mins'],
                cook_time_mins=recipe['cook_time_mins'],
                servings=recipe['servings'],
            )
            db.session.merge(new)
            db.session.commit()

            return render_template('search_results.html', query=query, results=[_recipe_card(new)])

        # If not found in Spoonacular, try GPT
        image_path = None
        content = ''
        ingredients = []
        instructions = ''
        calories = None
        image_url = None

        try:
            print("Calling GPT-turbo for recipe generation...")
            if client is None:
                raise RuntimeError("OpenAI package is not installed.")
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
            content = _openai_message_text(resp)
            if not content:
                raise ValueError(
                    "OpenAI returned empty recipe text (refusal, empty completion, or API issue). "
                    "Check OPENAI_API_KEY and try again."
                )

            flash("✅ Generated by GPT-turbo", "warning")

            if os.getenv('SERPER_API_KEY'):
                try:
                    conn = http_client.HTTPConnection('google.serper.dev')
                    payload = json.dumps({"q": query})
                    headers = {
                        'X-API-KEY': os.getenv('SERPER_API_KEY'),
                        'Content-Type': 'application/json',
                    }
                    conn.request("POST", "/search", payload, headers)
                    conn.getresponse().read()
                except Exception as serper_err:
                    print(f"Serper step skipped: {serper_err}")

            cleaned = _strip_markdown_json_fences(content)
            recipe_data = json.loads(cleaned)
            ingredients, instructions, calories = _normalize_gpt_recipe_payload(recipe_data)

            try:
                response_image_gen = client.images.generate(
                    prompt=f"Recipe image for {query}",
                    n=1,
                    size="1024x1024",
                )
                image_url = response_image_gen.data[0].url
            except Exception as img_err:
                print(f"OpenAI image generation skipped: {img_err}")
                image_url = None

            try:
                downloader.download(
                    query,
                    limit=1,
                    output_dir='bing_images',
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=60,
                )
                image_path_list = glob.glob(f'bing_images/{query}/*')
                image_path = image_path_list[0] if image_path_list else None
                if image_path:
                    upload_result = cloudinary.uploader.upload(
                        image_path,
                        folder="recipes",
                        public_id=f"{query}_image",
                    )
                    image_url = upload_result['secure_url']
            except Exception as dl_err:
                print(f"Bing / Cloudinary download step skipped: {dl_err}")

        except Exception as gen_err:
            print(f"GPT recipe fallback used: {gen_err}")
            ingredients = []
            instructions = (content or "").strip() or f"Simple cooking notes for {query}: prepare your ingredients, cook until done, season to taste, and serve warm."
            calories = 350
            image_url = None
        finally:
            _cleanup_bing_download_dir(query, image_path)

        new = Recipe(
            name=query,
            ingredients=json.dumps(ingredients),
            instructions=instructions,
            calories=calories,
            image_url=image_url,
            category='Main',
            cuisine='Global',
            diet_type='Regular',
            prep_time_mins=10,
            cook_time_mins=30,
            servings=4,
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
        results.append(_recipe_card(new))

        return render_template('search_results.html', query=query, results=results)

    except Exception as e:
        print(f"❌ Error: {e}")
        flash(f"Error searching recipes: {str(e)}")
        return redirect(url_for('home'))


@app.route('/api/favorites', methods=["GET"])
@login_required
def api_favorites():
    favorites = db.session.query(Recipe).join(Favorites, Favorites.RecipeId == Recipe.id).filter(Favorites.UserId == current_user.id).all()
    return jsonify([_recipe_card(recipe) for recipe in favorites])

@app.route('/api/favorites/toggle', methods=['POST'])
@login_required
def toggle_favorite():
    recipe_id = request.form.get('recipe_id') or (request.get_json(silent=True) or {}).get('recipe_id')
    recipe_id = _int_or_none(recipe_id)
    if not recipe_id:
        return jsonify({'success': False, 'message': 'recipe_id is required'}), 400
    favorite = Favorites.query.filter_by(UserId=current_user.id, RecipeId=recipe_id).first()
    if favorite:
        db.session.delete(favorite)
        favorited = False
    else:
        db.session.add(Favorites(UserId=current_user.id, RecipeId=recipe_id))
        favorited = True
    db.session.commit()
    return jsonify({'success': True, 'favorited': favorited})

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.bio = request.form.get('bio', '').strip()
        current_user.dietary_pref = request.form.get('dietary_pref', '').strip()
        current_user.avatar_url = request.form.get('avatar_url', '').strip()
        db.session.commit()
        flash('Profile updated.', 'success')
        return redirect(url_for('profile'))
    favorites = db.session.query(Recipe).join(Favorites, Favorites.RecipeId == Recipe.id).filter(Favorites.UserId == current_user.id).all()
    submitted = Recipe.query.filter_by(submitted_by=current_user.id).order_by(Recipe.id.desc()).all()
    return render_template('profile.html', favorites=[_recipe_card(r) for r in favorites], submitted=[_recipe_card(r) for r in submitted])

@app.route('/submit-recipe', methods=['GET', 'POST'])
@login_required
def submit_recipe():
    if request.method == 'POST':
        recipe = Recipe(
            name=request.form.get('name', '').strip(),
            ingredients=json.dumps([line.strip() for line in request.form.get('ingredients', '').splitlines() if line.strip()]),
            instructions=request.form.get('instructions', '').strip(),
            image_url=request.form.get('image_url', '').strip() or None,
            calories=_int_or_none(request.form.get('calories')),
            category=request.form.get('category', '').strip() or 'Main',
            cuisine=request.form.get('cuisine', '').strip() or 'Global',
            diet_type=request.form.get('diet_type', '').strip() or 'Regular',
            prep_time_mins=_int_or_none(request.form.get('prep_time_mins')),
            cook_time_mins=_int_or_none(request.form.get('cook_time_mins')),
            servings=_int_or_none(request.form.get('servings')) or 4,
            protein_g=_int_or_none(request.form.get('protein_g')),
            carbs_g=_int_or_none(request.form.get('carbs_g')),
            fat_g=_int_or_none(request.form.get('fat_g')),
            fiber_g=_int_or_none(request.form.get('fiber_g')),
            submitted_by=current_user.id,
            is_user_submitted=True,
        )
        if not recipe.name or not recipe.ingredients or not recipe.instructions:
            flash('Recipe name, ingredients, and instructions are required.', 'danger')
            return render_template('submit_recipe.html')
        db.session.add(recipe)
        db.session.commit()
        flash('Your recipe is live.', 'success')
        return redirect(url_for('view_recipe', recipe_id=recipe.id))
    return render_template('submit_recipe.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            token = secrets.token_urlsafe(32)
            db.session.add(PasswordResetToken(user_id=user.id, token=token, expires_at=datetime.utcnow() + timedelta(hours=1)))
            db.session.commit()
            try:
                _send_reset_email(user, token)
                flash('A reset link has been sent to your email.', 'success')
            except Exception as exc:
                print(f"Password reset email failed: {exc}")
                flash(f'Reset link created: {url_for("reset_password", token=token)}', 'warning')
        else:
            flash('If that email exists, a reset link has been sent.', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html', form=form)

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    reset = PasswordResetToken.query.filter_by(token=token, used=False).first_or_404()
    if reset.expires_at < datetime.utcnow():
        flash('This reset link has expired.', 'danger')
        return redirect(url_for('forgot_password'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        reset.user.set_password(form.password.data)
        reset.used = True
        db.session.commit()
        flash('Your password has been updated.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form)




    



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
        try:
            spoon_res = requests.get(spoon_url, params=spoon_params, timeout=12)
            spoon_data = spoon_res.json().get('results', [])
        except requests.RequestException as exc:
            print(f"Spoonacular API search skipped: {exc}")
            spoon_data = []

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

        image_path = None
        content = ''
        ingredients = []
        instructions = ''
        calories = None
        image_url = None

        try:
            print("Calling GPT-turbo for recipe generation...")
            if client is None:
                raise RuntimeError("OpenAI package is not installed.")
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
            content = _openai_message_text(resp)
            if not content:
                raise ValueError(
                    "OpenAI returned empty recipe text (refusal, empty completion, or API issue). "
                    "Check OPENAI_API_KEY and try again."
                )

            if os.getenv('SERPER_API_KEY'):
                try:
                    conn = http_client.HTTPConnection('google.serper.dev')
                    payload = json.dumps({"q": query})
                    headers = {
                        'X-API-KEY': os.getenv('SERPER_API_KEY'),
                        'Content-Type': 'application/json',
                    }
                    conn.request("POST", "/search", payload, headers)
                    conn.getresponse().read()
                except Exception as serper_err:
                    print(f"Serper step skipped: {serper_err}")

            cleaned = _strip_markdown_json_fences(content)
            recipe_data = json.loads(cleaned)
            ingredients, instructions, calories = _normalize_gpt_recipe_payload(recipe_data)

            try:
                response_image_gen = client.images.generate(
                    prompt=f"Recipe image for {query}",
                    n=1,
                    size="1024x1024",
                )
                image_url = response_image_gen.data[0].url
            except Exception as img_err:
                print(f"OpenAI image generation skipped: {img_err}")
                image_url = None

            try:
                downloader.download(
                    query,
                    limit=1,
                    output_dir='bing_images',
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=60,
                )
                image_path_list = glob.glob(f'bing_images/{query}/*')
                image_path = image_path_list[0] if image_path_list else None
                if image_path:
                    upload_result = cloudinary.uploader.upload(
                        image_path,
                        folder="recipes",
                        public_id=f"{query}_image",
                    )
                    image_url = upload_result['secure_url']
            except Exception as dl_err:
                print(f"Bing / Cloudinary download step skipped: {dl_err}")

        except Exception as gen_err:
            print(f"GPT API recipe fallback used: {gen_err}")
            ingredients = []
            instructions = (content or "").strip() or f"Simple cooking notes for {query}: prepare your ingredients, cook until done, season to taste, and serve warm."
            calories = 350
            image_url = None
        finally:
            _cleanup_bing_download_dir(query, image_path)

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
        dishes = famous_dish_cards_from_labels(5)
        return render_template('famous_dishes.html', dishes=dishes)
    except Exception as e:
        print(f"Error fetching famous dishes: {e}")
        flash("Error loading famous dishes", "error")
        return redirect(url_for('home'))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- REVISED PREDICT_DISH FUNCTION ---
def predict_dish(image_data, top_k=3):
    """
    Predicts the dish from the given image data using the loaded model.
    Args:
        image_data: Bytes of the image file.
        top_k: Number of top predictions to return (default 3).
    Returns:
        tuple: (predictions_list, error_message)
        predictions_list: [{dish, confidence}, ...] sorted by confidence desc
    """
    global chefly_model, class_labels_list

    # Ensure model and labels are loaded
    if chefly_model is None or class_labels_list is None:
        success, error = load_chef_model_and_labels()
        if not success:
            return None, error

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
            return None, "Mismatch between model output and class_labels.json. Model output shape might be incorrect."

        top_k = max(1, min(int(top_k), len(class_labels_list)))
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_predictions = [
            {
                'dish': class_labels_list[i],
                'confidence': float(predictions[i]),
            }
            for i in top_indices
        ]

        best = top_predictions[0]
        print(f"\n✅ Predicted: {best['dish']} ({best['confidence']*100:.2f}% confidence)\n")
        print(f"🔝 Top {top_k} Predictions:")
        for entry in top_predictions:
            print(f"{entry['dish']}: {entry['confidence'] * 100:.2f}%")

        return top_predictions, None

    except Exception as e:
        print("Error making prediction:", e)
        return None, str(e)
# --- END REVISED PREDICT_DISH FUNCTION ---


@app.route('/api/trained', methods=["GET"])
def api_trained_dishes():
    return jsonify(trained_dish_rows_from_labels()), 200


@app.route('/trained_dishes', methods=["GET"])
def trained_dishes():
    dishes = trained_dish_rows_from_labels()
    return render_template('trained_dishes.html', dishes=dishes)

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

        # 4. Perform prediction (top 3)
        top_predictions, prediction_error = predict_dish(image_data, top_k=3)

        if prediction_error:
            return jsonify({'error': f"Prediction failed: {prediction_error}"}), 500

        # 5. Upload image to Cloudinary (only if prediction was successful)
        try:
            upload_result = cloudinary.uploader.upload(io.BytesIO(image_data), resource_type="image")
            image_url = upload_result['secure_url']
        except Exception as e:
            return jsonify({'error': f"Failed to upload image to Cloudinary: {str(e)}"}), 500

        best = top_predictions[0]
        # 6. Return successful response
        return jsonify({
            'url': image_url,
            'predicted_dish': best['dish'],
            'confidence': best['confidence'],
            'predictions': top_predictions,
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
        rating, rating_count = _avg_rating(recipe.id)
        reviews = Rating.query.filter_by(recipe_id=recipe.id).order_by(Rating.created_at.desc()).all()
        is_favorite = False
        if current_user.is_authenticated:
            is_favorite = Favorites.query.filter_by(UserId=current_user.id, RecipeId=recipe.id).first() is not None
        return render_template('recipe.html', recipe=recipe, rating=rating, rating_count=rating_count, reviews=reviews, is_favorite=is_favorite)

    except Exception as e:
        flash(f"Error loading recipe: {str(e)}", "error")
        return redirect(url_for('home'))

@app.route('/recipe/<int:recipe_id>/rate', methods=['POST'])
@login_required
def rate_recipe(recipe_id):
    recipe = Recipe.query.get_or_404(recipe_id)
    score = _int_or_none(request.form.get('score'))
    if score is None or score < 1 or score > 5:
        flash('Please choose a rating from 1 to 5 stars.', 'danger')
        return redirect(url_for('view_recipe', recipe_id=recipe.id))
    rating = Rating.query.filter_by(user_id=current_user.id, recipe_id=recipe.id).first()
    if not rating:
        rating = Rating(user_id=current_user.id, recipe_id=recipe.id)
        db.session.add(rating)
    rating.score = score
    rating.comment = request.form.get('comment', '').strip()
    db.session.commit()
    flash('Thanks for reviewing this recipe.', 'success')
    return redirect(url_for('view_recipe', recipe_id=recipe.id))

@app.route('/admin/analytics')
@login_required
def admin_analytics():
    stats = {
        'users': User.query.count(),
        'recipes': Recipe.query.count(),
        'user_recipes': Recipe.query.filter_by(is_user_submitted=True).count(),
        'favorites': Favorites.query.count(),
        'reviews': Rating.query.count(),
    }
    top_recipes = db.session.query(Recipe, func.count(Favorites.RecipeId).label('favorite_count')).outerjoin(Favorites, Favorites.RecipeId == Recipe.id).group_by(Recipe.id).order_by(func.count(Favorites.RecipeId).desc()).limit(8).all()
    top_rated = db.session.query(Recipe, func.avg(Rating.score).label('avg_score')).join(Rating, Rating.recipe_id == Recipe.id).group_by(Recipe.id).order_by(func.avg(Rating.score).desc()).limit(8).all()
    return render_template('admin_analytics.html', stats=stats, top_recipes=top_recipes, top_rated=top_rated)
    
from flask import jsonify

@app.route('/api/recipe/<int:recipe_id>', methods=['GET'])
def api_view_recipe(recipe_id):
    try:
        recipe = Recipe.query.get(recipe_id)
        if not recipe:
            return jsonify({"error": "Recipe not found"}), 404

        # Clean up ingredient list
        ingredients_raw = recipe.ingredients
        if isinstance(ingredients_raw, str):
            ingredients = [ing.strip().strip('"').strip("'") for ing in ingredients_raw.split(',')]
        else:
            ingredients = ingredients_raw

        return jsonify({
            "id": recipe.id,
            "name": recipe.name,
            "ingredients": ingredients,
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

@app.context_processor
def template_helpers():
    return {'placeholder_dish_image': placeholder_dish_image}

@app.route('/start_search', methods=['POST'])
def start_search():
    query = request.form.get('q', '')
    return render_template('loading_page.html', query=query)

if __name__ == '__main__':
    # Initialize the database
    with app.app_context():
        db.create_all()
        seed_recipe_metadata()
        
    app.run(host='0.0.0.0', port=8080, debug=True)
