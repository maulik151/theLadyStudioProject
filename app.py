from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, AdminIndexView
from flask_admin.contrib.sqla import ModelView
from models import db, User, Product, AddToCart, UserInteraction
from werkzeug.security import generate_password_hash, check_password_hash
from flask_admin.menu import MenuLink
from flask_wtf.file import FileField, FileAllowed
import base64
import os
from wtforms import StringField, IntegerField
from flask_wtf import FlaskForm
from flask import send_from_directory, session, jsonify
from sqlalchemy.orm import join
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
import numpy as np

from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


class LoginMenuLink(MenuLink):
    def is_accessible(self):
        return not current_user.is_authenticated 

class LogoutMenuLink(MenuLink):
    def is_accessible(self):
        return current_user.is_authenticated 

class MyAdminIndexView(AdminIndexView):
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_superuser
    
    def inaccessible_callback(self, name, **kwargs):
        # Redirect to login page if user doesn't have access
        return redirect(url_for('login'))

class UserModelView(ModelView):
    column_exclude_list = ['password']

    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_superuser

class ProductForm(FlaskForm):
    gender = StringField('Gender')
    masterCategory = StringField('Master Category')
    subCategory = StringField('Sub Category')
    articleType = StringField('Article Type')
    baseColour = StringField('Base Colour')
    season = StringField('Season')
    year = IntegerField('Year')
    usageDescription = StringField('Usage Description')
    productDisplayName = StringField('Product Display Name')
    productImage = FileField('Product Image', validators=[FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])
    price = IntegerField('Price')

# Customized product model view    
class ProductModelView(ModelView):
    column_list = ('id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usageDescription', 'productDisplayName', 'price')

    form_extra_fields = {
        'productImage': FileField('Product Image', validators=[FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])
    }

    def on_model_change(self, form, model, is_created):
        if form.productImage.data is not None and hasattr(form.productImage.data, 'filename') and form.productImage.data.filename:
            # If a new image is uploaded, update the productImage field
            model.productImage = form.productImage.data.read()
        else:
            # If no new image is uploaded or if the image field is empty, retain the existing productImage value
            model.productImage = model.productImage

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:rootadmin@localhost/project'

# Initialize Flask extensions
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

admin = Admin(app, index_view=MyAdminIndexView())
admin.add_view(UserModelView(User, db.session))
admin.add_view(ProductModelView(Product, db.session))
admin.add_link(LogoutMenuLink(name='Logout', category='', url="/logout"))
admin.add_link(LoginMenuLink(name='Login', category='', url="/login"))

# Load the trained model
model = load_model('trained_model.h5')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Add a route to serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

def getProductImage(id):
    product = Product.query.get(id)  # Fetch the product with id 2188
    product_image = product.productImage  # Get the product image data
    img_filename = str(id) + ".jpg"

    # Write the image to the static directory
    imgPath = os.path.join(app.root_path, "static", "images", img_filename)
    with open(imgPath, "wb") as f:
        f.write(product_image)
    
    return img_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    # Check if the user is logged in
    logged_in = current_user.is_authenticated

    # If the user is logged in, get the username
    username = session.get('username') if logged_in else None

    # Get the page number from the request
    page = request.args.get('page', 1, type=int)
    per_page = 12
    offset = (page - 1) * per_page
    products = Product.query.limit(page * per_page).all()
    
    message  = None
    # Check if the orderConfirmed value is received
    order_confirmed = request.args.get('orderConfirmed')
    if order_confirmed:
        message = f"Order for {username} has been placed."
        flash(message, 'success')  # Flash the success message
        return redirect(url_for('index'))

    product_details = []
    for product in products:
        id = product.id
        img_filename = getProductImage(id)
        product_display_name = product.productDisplayName
        price = product.price
        product_details.append({'id': id, 'img_filename': img_filename, 'product_display_name': product_display_name, 'price': price})

    return render_template('index.html', product_details=product_details, page=page, username=username, logged_in=logged_in, message  = message)


@app.route('/add_to_cart/<int:product_id>/<source_page>', methods=['POST'])
@login_required 
def add_to_cart(product_id, source_page):
    if request.method == 'POST':
        quantity = int(request.form.get('quantity'))  # Retrieve quantity from the form
        price = float(request.form.get('price'))  # Retrieve price from the form
        
        # Create a new entry in the AddToCart table
        cart_item = AddToCart(username=current_user.username, productid=product_id, quantity=quantity, price=price)
        db.session.add(cart_item)
        db.session.commit()
        
        flash('Product ' + str(product_id) + ' added to cart successfully!', 'cart')  # Flash the cart message with 'cart' category
        
        if source_page == 'index':
            return redirect(url_for('index', added_to_cart=True))  # Redirect user to the homepage with added_to_cart flag
        elif source_page == 'singleProduct':
            return redirect(url_for('singleProduct', product_id=product_id, added_to_cart=True))  # Redirect user to the product page
    
    return redirect(url_for('index'))  # Redirect user to the homepage if the request method is not POST


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def get_recommendations(img_path):
    features = feature_extraction(img_path, model)
    indices = recommend(features, feature_list)
    print("recommended_products::::", indices)
    recommended_images = [filenames[idx] for idx in indices[0]]
# Extract filenames and filter those ending with ".jpg"
    jpg_filenames = [os.path.splitext(os.path.basename(file_path))[0] for file_path in recommended_images if file_path.endswith(".jpg")]
    return jpg_filenames

@app.route('/singleProduct/<int:product_id>')
def singleProduct(product_id):
    # Fetch the product image filename and product with id
    img_filename = getProductImage(product_id)
    product = Product.query.get(product_id)

    # Generate image path and get recommendations
    img_path = './static/images/' + img_filename

    recommendations = get_recommendations(img_path)

    # Fetch matching products and process their image filenames
    matching_products = Product.query.filter(Product.id.in_(recommendations)).all()
    matching_product_ids = [product.id for product in matching_products]
    matching_product_images = {product.id: getProductImage(product.id) for product in matching_products}

    # Pass data to the template
    return render_template('singleProduct.html',
                        img_filename=img_filename,
                        product=product,
                        matching_products=matching_products,
                        matching_product_images=matching_product_images
                        )

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/cart')
@login_required 
def cart():
    cart_items = db.session.query(AddToCart, Product).\
        outerjoin(Product, AddToCart.productid == Product.id).\
        filter(AddToCart.username == current_user.username).all()
    
    # Extract product names from the joined query
    cart_items_with_names = [(add_to_cart, product.productDisplayName) for add_to_cart, product in cart_items]
    
    total_sum = 0

    for add_to_cart, product in cart_items:
        subtotal = add_to_cart.quantity * product.price
        total_sum += subtotal

    return render_template('cart.html', cart_items=cart_items_with_names, total_sum = total_sum, logged_in=True, username = current_user.username)

@app.route('/remove_item/<int:item_id>', methods=['GET'])
@login_required
def remove_item(item_id):
    # Logic to remove the item from the cart based on the item_id
    # Example:
    cart_item = AddToCart.query.get(item_id)
    db.session.delete(cart_item)
    db.session.commit()
    
    # Redirect the user back to the cart page after removing the item
    return redirect(url_for('cart'))


@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

@app.route('/shop')
def shop():
    return render_template('shop.html')

@app.route('/checkout')
def checkout():
    cart_items = db.session.query(AddToCart, Product).\
        outerjoin(Product, AddToCart.productid == Product.id).\
        filter(AddToCart.username == current_user.username).all()
    
    # Extract product names from the joined query
    cart_items_with_names = [(add_to_cart, product.productDisplayName) for add_to_cart, product in cart_items]
    total_sum = 0

    for add_to_cart, product in cart_items:
        subtotal = add_to_cart.quantity * product.price
        total_sum += subtotal

    return render_template('checkout.html', cart_items=cart_items_with_names, total_sum = total_sum, logged_in=True, username = current_user.username)

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        first_name = request.form['firstname']
        last_name = request.form['lastname']
        email = request.form['email']
        address = request.form['address']
        password = request.form['password']
        confirm_password = request.form['confirmpassword']

        
        # Check if password and confirm_password match
        if password != confirm_password:
            return 'Passwords do not match'

        # Check if username already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('signup.html', user_exists=True, email=email)
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, first_name=first_name, last_name=last_name, address=address, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash(f'{username} registered successfully!', 'success')  # Pass the username to the flash message

        return render_template('login.html', registration_success=True, username=username)  # Pass registration success flag and username
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']
        # user = User.get(username)
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            session['username'] = username
            if user.is_superuser:
                return redirect(url_for('admin.index'))
            else:
                page = 1
                per_page = 12
                # offset = (page - 1) * per_page
                products = Product.query.limit(page * per_page).all()

                product_details = []
                for product in products:
                    id = product.id
                    img_filename = getProductImage(id)
                    product_display_name = product.productDisplayName
                    price = product.price
                    product_details.append({'id': id, 'img_filename': img_filename, 'product_display_name': product_display_name, 'price': price})

                return render_template('index.html', username=username, logged_in=True, product_details=product_details, page=1)
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    session['username'] = ''
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
