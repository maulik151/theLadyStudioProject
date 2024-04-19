from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import CheckConstraint

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(255), nullable=False)
    last_name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Use 'password' attribute
    is_superuser = db.Column(db.Boolean, default=False)  # Boolean field for superuser status

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)
    
class Product(db.Model):
    __tablename__ = 'product'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    gender = db.Column(db.String(20))
    masterCategory = db.Column(db.String(50))
    subCategory = db.Column(db.String(50))
    articleType = db.Column(db.String(50))
    baseColour = db.Column(db.String(50))
    season = db.Column(db.String(20))
    year = db.Column(db.Integer)
    usageDescription = db.Column(db.String(255))
    productDisplayName = db.Column(db.String(100))
    productImage = db.Column(db.LargeBinary, nullable=False)
    price = db.Column(db.Integer, nullable=False, default=0)


class AddToCart(db.Model):
    __tablename__ = 'addtocart'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), nullable=False)
    productid = db.Column(db.Integer, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.DECIMAL(10, 2), nullable=False)

class UserInteraction(db.Model):
    __tablename__ = 'userInteraction'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), nullable=False)
    productid = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Integer, CheckConstraint('rating >= 0 AND rating <= 5'), default=0, nullable=True)
    productclickcount = db.Column(db.Integer, default=0)