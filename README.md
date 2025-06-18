Chefly - AI-Powered Cooking Assistant


**Chefly** is a modern cooking web application that leverages Convolutional Neural Networks (CNN) to recognize food
images and provide users with detailed recipes, ingredients, and calorie information. The app also offers robust
text-based search and integrates with external APIs for comprehensive recipe coverage.


**Trained Dishes:**
"Ackee_and_Saltfish", "Adobo", "Aloo_Keema", "Arepas", "Asado",
"Baklava", "Banh_Mi", "Bhindi_Masala", "Bibimbap", "Biryani",
"Bistecca_alla_Fiorentina", "Boeuf_Bourguignon", "Borscht", "Bratwurst", "Burgers",
"Ceviche", "Cioppino", "Clam_Chowder", "Congee", "Coq_au_Vin",
"Couscous", "Croissant", "Curry", "Daal_Fry", "Dim_Sum",
"Dosa", "Empanadas", "Enchiladas", "Escargots", "Falafel",
"Feijoada", "Fish_and_Chips", "French_Fries", "Gazpacho", "Gelato",
"Goulash", "Gravlax", "Green_Curry", "Gumbo", "Gyoza",
"Gyros", "Haggis", "Haleem", "Hot_Dogs", "Hummus",
"Irish_Stew", "Pizza", "Ramen", "Sushi", "Takoyaki"


## - Features
- **AI-Driven Dish Recognition:** Upload an image of your dish (JPG, JPEG, PNG, or GIF) and let our CNN model
predict if it matches one of the 16 trained dishes. Get instant recipe details!
- **Text-Based Search:** Search for any dish by name. The app first checks your database, then queries Spoonacular
for more results, and finally uses OpenAI Turbo 3.5 as a fallback.
- **Recipe Details:** Includes step-by-step cooking instructions, ingredient lists, and calorie information for each dish.
- **User Accounts:** Secure login and registration endpoints.
- **Contact Support:** Dedicated endpoint for user feedback and support.

  
## - How It Works
1. **Image Upload**
- **Supported formats:** JPG, JPEG, PNG, GIF
- **Process:** The CNN model analyzes the uploaded image. If the dish is one of the 16 trained classes, the app
displays recipe details, ingredients, and calorie information.
- **If not recognized:** The app informs the user that the dish is not supported.
2. **Text-Based Search**
- **Search flow:**
1. **Local database:** The app checks if the dish exists in your database.
2. **Spoonacular API:** If not found, it queries Spoonacular for recipe details.
3. **OpenAI Turbo 3.5:** As a fallback, it uses OpenAI to generate a response (without an image).
- **Response:** Always provides recipe instructions and ingredients, with calories if available.
3. **User Management**
- **Login/Register:** Users can create accounts and log in securely.
- **Contact:** Users can send feedback or support requests.

  
## - Technology Stack
- **Backend:** Python (Flask/Django/FastAPI)
- **Machine Learning:** Convolutional Neural Network (CNN) for image recognition
- **APIs:** Spoonacular (recipe data), OpenAI Turbo 3.5 (fallback recipe generation)
- **Database:** Local database for storing user data and recipes
- **Frontend:** HTML and CSS

  
## - Architecture Overview
- **Image Encoder:** Pre-trained CNN MobileNetV2 fine-tuned on 16 food classes to extract features from images.
- **Recipe Generator:** For text-based search, integrates with local DB, Spoonacular, and OpenAI for recipe generation.
- **User Management:** Secure endpoints for login, registration, and contact.


## - Example Usage
1. Upload a dish image (e.g., pizza.jpg)
- If recognized, see recipe, ingredients, and calories.
- If not, get a message: "Dish not recognized."
2. Search for a dish by name (e.g., "chicken curry")
- If found in your database, see recipe details.
- If not, Spoonacular is queried.
- If still not found, OpenAI Turbo 3.5 generates a response.


## - Installation
*Instructions for installation and setup*
git clone https://github.com/saadJP23/chefly.git
cd chefly
pip install -r requirements.txt
python index.py


## - Key Points
- **CNN Model:** Trained on 16 distinct dishes for accurate image recognition.
- **Multi-Source Recipe Search:** Combines local database, Spoonacular, and OpenAI for comprehensive results.
- **User-Friendly:** Intuitive interface for both image and text-based queries.
---
*Chefly makes cooking smarter and more accessible with the power of AI!*
