import random
import uuid

import pandas as pd
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, func,TEXT, Float
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import relationship
from app import db, app
from datetime import datetime


class Faculty(db.Model):
    __tablename__ = 'faculty'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(String(255))
    users = relationship('User', backref='faculty', lazy=True)
    __table_args__ = {'extend_existing': True}


class User(db.Model):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(80), unique=False, nullable=False)
    email = Column(String(80), unique=False, nullable=False)
    phone_number = Column(String(20), nullable=False)
    gender = Column(Boolean(), default=False)
    age = Column(Integer)
    address = Column(String(100), nullable=True)
    enrollment_date = Column(DateTime, default=datetime.utcnow)
    ms_code = Column(String(10), unique=False, default='0')
    fs_uniquifier = Column(String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    faculty_id = Column(Integer, ForeignKey('faculty.id'), nullable=True)
    reviews = relationship('Review', backref="user", lazy=True)
    __table_args__ = {'extend_existing': True}

    def __str__(self):
        return self.full_name


class Book(db.Model):
    __tablename__ = 'book'
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    book_category = Column(String(255))
    star_rating = Column(Integer)
    price = Column(Float, nullable=False, default=0.0)
    stock = Column(String(30), nullable=False, default=0)
    quantity = Column(Integer, nullable=False, default=0)
    is_enable = Column(Boolean, default=True)
    create_at = Column(DateTime, default=func.now())
    reviews = relationship('Review', backref='book', lazy=True)
    __table_args__ = {'extend_existing': True}


class Borrowing_Receipt(db.Model):
    __tablename__ = 'borrowing_receipt'
    id = Column(Integer, primary_key=True, autoincrement=True)
    borrow_date = Column(DateTime, nullable=False)
    due_date = Column(DateTime, nullable=False)
    return_date = Column(DateTime)
    quantity_book = Column(Integer, nullable=False, default=0)
    quantity_return = Column(Integer, nullable=False, default=0)
    penalty = Column(String(255))
    amount = Column(Float)
    user_ms_code = Column(String(10), nullable=False)
    book_title = Column(String(255), nullable=False)
    __table_args__ = {'extend_existing': True}


class Review(db.Model):
    __tablename__ = 'review'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    book_id = Column(Integer, ForeignKey('book.id'), nullable=False)
    point = Column(Integer)
    __table_args__ = {'extend_existing': True}


if __name__ == "__main__":
    with app.app_context():
        db.create_all()


        # def convert_star_rating(rating_text):
        #     rating_map = {
        #         'One': 1,
        #         'Two': 2,
        #         'Three': 3,
        #         'Four': 4,
        #         'Five': 5
        #     }
        #     return rating_map.get(rating_text, None)
        #
        #
        # def import_books_from_excel(file_path):
        #     df = pd.read_excel(file_path)
        #
        #     for index, row in df.iterrows():
        #         star_rating = convert_star_rating(row['Star_rating'])
        #         if star_rating is None:
        #             continue
        #         existing_book = Book.query.filter_by(title=row['Title']).first()
        #         if existing_book:
        #             continue
        #         book = Book(
        #             title=row['Title'],
        #             book_category=row['Book_category'],
        #             star_rating=star_rating,
        #             price=row['Price'],
        #             stock=row['Stock'],
        #             quantity=row['Quantity']
        #         )
        #         db.session.add(book)
        #     db.session.commit()
        #
        #
        # file_path = 'C:/Users/MinhQuan/OneDrive/Desktop/DoAnNganh/DoAnNganh/app/static/data_import/Book.xlsx'
        # import_books_from_excel(file_path)
        #
        # def import_faculty_from_excel(path):
        #     file = pd.read_excel(path)
        #     for index, row in file.iterrows():
        #         faculty = Faculty(
        #             name=row['Khoa']
        #         )
        #         db.session.add(faculty)
        #     db.session.commit()
        #
        # file = 'C:/Users/MinhQuan/OneDrive/Desktop/DoAnNganh/DoAnNganh/app/static/data_import/Faculty.xlsx'
        # import_faculty_from_excel(file)
        #
        #
        # def import_User_from_excel(f):
        #     try:
        #         File = pd.read_excel(f)
        #         faculty_ids = [faculty.id for faculty in Faculty.query.all()]
        #
        #         for index, row in File.iterrows():
        #             # Kiểm tra xem 'full_name' hoặc 'ms_code' có bị thiếu hay không
        #             if pd.isna(row['full_name']) or pd.isna(row['code']):
        #                 print(f"Bỏ qua bản ghi vì thiếu full_name hoặc code: {row}")
        #                 continue
        #
        #             # Kiểm tra xem người dùng đã tồn tại hay chưa
        #             existing_user = User.query.filter_by(full_name=row['full_name'], ms_code=row['code']).first()
        #             if existing_user:
        #                 print(f"Người dùng đã tồn tại: {row['full_name']} với code: {row['code']}")
        #                 continue
        #
        #             # Tạo người dùng mới
        #             user = User(
        #                 full_name=row['full_name'],
        #                 email=row['email'],
        #                 phone_number=row['phone_number'],
        #                 gender=row['gender'],
        #                 age=row['age'],
        #                 address=row['address'],
        #                 ms_code=row['code'],
        #                 faculty_id=random.choice(faculty_ids) if faculty_ids else None
        #             )
        #             db.session.add(user)
        #
        #         db.session.commit()
        #         print('Đã thêm user thành công')
        #
        #     except IntegrityError as e:
        #         print(f"Đã xảy ra lỗi IntegrityError: {e}")
        #         db.session.rollback()
        #     except Exception as e:
        #         print(f"Đã xảy ra lỗi: {e}")
        #         db.session.rollback()
        #
        # f = 'C:/Users/MinhQuan/OneDrive/Desktop/DoAnNganh/DoAnNganh/app/static/data_import/User.xlsx'
        # import_User_from_excel(f)
        # print("đã thêm user")
        #
        # def add_reviews(path):
        #     try:
        #         df = pd.read_excel(path)
        #         print("Các cột có trong tập tin:", df.columns.tolist())
        #         if not {'user_id', 'book_id', 'rant'}.issubset(df.columns):
        #             raise ValueError("Tập tin Excel cần có các cột 'user_id', 'book_id', và 'rant'.")
        #         valid_user_ids = set([user.id for user in db.session.query(User.id).all()])
        #         valid_book_ids = set([book.id for book in db.session.query(Book.id).all()])
        #
        #         reviews = []
        #         for _, row in df.iterrows():
        #             user_id = row['user_id']
        #             book_id = row['book_id']
        #             point = row['rant']
        #             if user_id in valid_user_ids and book_id in valid_book_ids:
        #                 if db.session.query(Review).filter_by(user_id=user_id, book_id=book_id).first() is None:
        #                     reviews.append(Review(user_id=user_id, book_id=book_id, point=point))
        #                 else:
        #                     print(f"Review với book_id={book_id} và user_id={user_id} đã tồn tại.")
        #             else:
        #                 print(f"User ID {user_id} hoặc Book ID {book_id} không hợp lệ.")
        #         if reviews:
        #             try:
        #                 db.session.add_all(reviews)
        #                 db.session.commit()
        #             except IntegrityError as e:
        #                 db.session.rollback()
        #                 print(f"Lỗi khi thêm dữ liệu vào cơ sở dữ liệu: {e}")
        #         else:
        #             print("Không có dữ liệu mới để thêm vào cơ sở dữ liệu.")
        #
        #     except Exception as e:
        #         print(f"Đã xảy ra lỗi: {e}")
        #
        #
        # path = 'C:/Users/MinhQuan/OneDrive/Desktop/DoAnNganh/DoAnNganh/app/static/data_import/Review.xlsx'
        # add_reviews(path)
        #
        #
        # def import_Borrowing_from_excel(path_file):
        #     borrow = pd.read_excel(path_file)
        #     for index , row in borrow.iterrows():
        #         borrowing = Borrowing_Receipt(
        #             borrow_date=row['borrow_date'],
        #             due_date=row['due_date'],
        #             return_date=row['return_date'],
        #             quantity_book=row['quantity_book'],
        #             quantity_return=row['quantity_return'],
        #             penalty=row['penalty'],
        #             amount=row['amount'],
        #             user_ms_code=row['user_ms_code'],
        #             book_title=row['book_title'],
        #         )
        #         db.session.add(borrowing)
        #     db.session.commit()
        #
        # path_file = 'C:/Users/MinhQuan/OneDrive/Desktop/DoAnNganh/DoAnNganh/app/static/data_import/Borrowing.xlsx'
        # import_Borrowing_from_excel(path_file)
        #
        # print('đã thêm borrowing')
        #

