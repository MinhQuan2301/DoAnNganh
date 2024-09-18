import base64
import io
import json
import os
import pickle

import numpy as np
from flask_admin import BaseView, expose
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from app import admin
from app.models import *
from flask_admin.contrib.sqla import ModelView
import matplotlib.pyplot as plt
from flask import render_template, request


class ProductView(ModelView):
    can_view_details = True
    can_export = True
    column_filters = ['title', 'book_category', 'star_rating']
    column_searchable_list = ['title', 'book_category', 'star_rating']
    column_formatters = {
        'create_at': lambda v, c, m, p: m.create_at.strftime('%d-%m-%Y') if m.create_at else ''
    }


class FacultyView(ModelView):
    can_view_details = True
    can_export = True
    column_filters = ['name']
    column_searchable_list = ['name']


class UserView(ModelView):
    can_view_details = True
    can_export = True
    column_exclude_list = ['fs_uniquifier']
    column_filters = ['email', 'full_name']
    column_searchable_list = ['full_name', 'email']


class ReviewModel(ModelView):
    column_filters = ['user_id', 'book_id', 'user_id']
    can_view_details = True
    can_export = True
    column_searchable_list = ['user_id', 'book_id', 'point']
    column_list = ('user_id', 'book_id', 'point')
    column_formatters = {
        'user': lambda v, c, m, p: m.user.username if m.user else 'N/A',
        'book': lambda v, c, m, p: m.book.title if m.book else 'N/A',
    }
    column_labels = {
        'User_Id': 'User',
        'Book_id': 'Book',
        'point': 'Point',
    }


class BorrowingModel(ModelView):
    can_view_details = True
    can_export = True
    column_filters = ['user_ms_code', 'book_title', 'amount', 'penalty', 'book_title']
    column_searchable_list = ['user_ms_code', 'book_title', 'amount', 'penalty', 'book_title']
    column_formatters = {
        'borrow_date': lambda v, c, m, p: m.borrow_date.strftime('%d-%m-%Y') if m.borrow_date else '',
        'due_date': lambda v, c, m, p: m.due_date.strftime('%d-%m-%Y') if m.due_date else '',
        'return_date': lambda v, c, m, p: m.return_date.strftime('%d-%m-%Y') if m.return_date else ''

    }


class ChartView(BaseView):
    @expose('/')
    def index(self):
        data = self.get_book_statistics()
        return self.render('admin/chart.html', chart_data=data)

    def get_book_statistics(self):
        stats_by_category = db.session.query(
            Book.book_category,
            func.count(Book.id).label('book_count')
        ).group_by(Book.book_category).all()
        stats_by_rating = db.session.query(
            Book.star_rating,
            func.count(Book.id).label('book_count')
        ).group_by(Book.star_rating).all()
        categories = [stat.book_category for stat in stats_by_category]
        category_counts = [stat.book_count for stat in stats_by_category]
        star_ratings = [stat.star_rating for stat in stats_by_rating]
        star_rating_counts = [stat.book_count for stat in stats_by_rating]
        return {
            'categories': categories,
            'category_counts': category_counts,
            'star_ratings': star_ratings,
            'star_rating_counts': star_rating_counts
        }


class TotalAmount(BaseView):
    @expose('/')
    def index(self):
        chart_data = self.borrowing_amount_by_User()
        penalty_data = self.penalty_by_month_2023()
        return self.render('admin/total_amount.html', chart_data=json.dumps(chart_data), penalty_data=json.dumps(penalty_data))

    def borrowing_amount_by_User(self):
        results = db.session.query(User.full_name, func.sum(Borrowing_Receipt.amount).label('total_amount')) \
            .join(Borrowing_Receipt, User.ms_code == Borrowing_Receipt.user_ms_code) \
            .group_by(User.full_name) \
            .all()
        data = {
            'labels': [result.full_name for result in results],
            'datasets': [{
                'label': 'Total Amount (VND)',
                'data': [result.total_amount for result in results],
                'backgroundColor': 'skyblue'
            }]
        }

        return data

    def penalty_by_month_2023(self):
        # Truy vấn để đếm số lượng từng loại hình phạt theo tháng trong năm 2023
        results = db.session.query(
            func.extract('month', Borrowing_Receipt.return_date).label('month'),
            Borrowing_Receipt.penalty,
            func.count(Borrowing_Receipt.penalty).label('penalty_count')
        ).filter(
            func.extract('year', Borrowing_Receipt.return_date) == 2023,
            Borrowing_Receipt.penalty.isnot(None)
        ).group_by(
            func.extract('month', Borrowing_Receipt.return_date),
            Borrowing_Receipt.penalty
        ).order_by(
            func.extract('month', Borrowing_Receipt.return_date),
            Borrowing_Receipt.penalty
        ).all()

        # Chuẩn bị dữ liệu cho biểu đồ
        months = [f"Tháng {int(month)}" for month in range(1, 13)]
        penalty_types = list(set(result.penalty for result in results))

        # Khởi tạo cấu trúc dữ liệu cho biểu đồ
        dataset = {penalty: [0] * 12 for penalty in penalty_types}

        # Điền dữ liệu vào cấu trúc
        for result in results:
            month_index = int(result.month) - 1
            penalty_type = result.penalty
            dataset[penalty_type][month_index] = result.penalty_count

        # Tạo cấu trúc dữ liệu cho Chart.js
        data = {
            'labels': months,
            'datasets': [
                {
                    'label': penalty_type,
                    'data': dataset[penalty_type],
                    'backgroundColor': self.get_color_for_penalty(penalty_type)
                }
                for penalty_type in penalty_types
            ]
        }

        return data

    def get_color_for_penalty(self, penalty_type):
        # Trả về màu sắc cho từng loại hình phạt (có thể tùy chỉnh)
        colors = {
            'Late Fee': 'red',
            'Damage Fee': 'blue',
            'Lost Book Fee': 'green'
        }
        return colors.get(penalty_type, 'gray')


class ForecastView(BaseView):
    @expose('/')
    def index(self):
        with app.app_context():
            # Truy xuất dữ liệu từ cơ sở dữ liệu
            engine = db.get_engine()  # Sử dụng db.get_engine() thay vì db.session.bind
            books = pd.read_sql_query('SELECT * FROM book', engine)
            reviews = pd.read_sql_query('SELECT * FROM review', engine)

            # Kết hợp dữ liệu từ các bảng
            book_reviews = pd.merge(books, reviews, left_on='id', right_on='book_id')

            # Tiền xử lý dữ liệu
            book_reviews['star_rating'] = book_reviews['star_rating'].fillna(0)
            book_reviews['point'] = book_reviews['point'].fillna(book_reviews['point'].mean())

            # Xây dựng mô hình
            X = book_reviews[['price', 'quantity', 'star_rating']]
            y = book_reviews['point']

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            book_reviews['predicted_point'] = model.predict(X)

            # Tạo biểu đồ
            plt.figure(figsize=(10, 6))
            plt.hist(book_reviews['point'], bins=30, alpha=0.5, label='Actual Points')
            plt.hist(book_reviews['predicted_point'], bins=30, alpha=0.5, label='Predicted Points')
            plt.legend(loc='upper right')
            plt.xlabel('Points')
            plt.ylabel('Frequency')
            plt.title('Distribution of Actual and Predicted Points')

            # Lưu biểu đồ vào một đối tượng BytesIO
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()

            # Chuyển đổi biểu đồ thành base64 để hiển thị trong HTML
            img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

            # Render biểu đồ
            return self.render('admin/forecast.html', img_data=img_base64)


class BookAnalysisView(BaseView):
    @expose('/')
    def index(self):
        # Lấy dữ liệu mượn sách cho năm 2023
        borrowing_data = self.get_annual_borrowing_data(year=2023)

        # Tổng hợp số lượng mượn theo sách
        aggregated_data = self.aggregate_annual_borrowing(borrowing_data)

        # Lấy tất cả sách từ bảng Book
        all_books = self.get_all_books()

        # Phân tích sách nên hủy và nhập thêm dựa trên tỉ lệ mượn
        books_to_remove, books_to_add = self.analyze_books(all_books, aggregated_data)

        # Trả kết quả về cho trang quản trị
        return self.render('admin/book_analysis.html',
                           aggregated_data=aggregated_data,
                           books_to_remove=books_to_remove,
                           books_to_add=books_to_add)

    def get_annual_borrowing_data(self, year=2023):
        # Truy vấn lấy tất cả các sách đã mượn trong năm 2023
        data = db.session.query(
            Borrowing_Receipt.book_title,
            func.sum(Borrowing_Receipt.quantity_book).label('total_borrowed')
        ).filter(
            func.extract('year', Borrowing_Receipt.borrow_date) == year
        ).group_by(
            Borrowing_Receipt.book_title
        ).all()

        # Trả về danh sách dữ liệu mượn
        return data

    def aggregate_annual_borrowing(self, borrowing_data):
        aggregated_data = {}

        for book_title, total_borrowed in borrowing_data:
            if book_title not in aggregated_data:
                aggregated_data[book_title] = 0
            aggregated_data[book_title] += total_borrowed

        return aggregated_data

    def get_all_books(self):
        # Truy vấn tất cả sách từ bảng Book
        books = db.session.query(Book.title, Book.quantity).all()
        return books

    def analyze_books(self, all_books, aggregated_data):
        books_to_remove = []
        books_to_add = []

        for book in all_books:
            book_title = book.title
            book_quantity = book.quantity
            total_borrowed = aggregated_data.get(book_title, 0)

            # Xác định tỉ lệ mượn sách
            borrow_ratio = total_borrowed / book_quantity if book_quantity > 0 else 0

            # Đưa ra quyết định dựa trên tỉ lệ mượn
            if total_borrowed == 0:  # Sách không được mượn lần nào
                books_to_remove.append((book_title, total_borrowed))
            elif borrow_ratio < 0.1:  # Tỉ lệ mượn thấp hơn 10% so với số lượng
                books_to_remove.append((book_title, total_borrowed))
            elif borrow_ratio > 0.5:  # Tỉ lệ mượn cao hơn 50% so với số lượng
                books_to_add.append((book_title, total_borrowed))

        return books_to_remove, books_to_add


class ModelView(BaseView):
    @expose('/')
    def index(self):
        return self.render('admin/model.html')

    @expose('/train', methods=['POST'])
    def train(self):
        try:
            # Huấn luyện mô hình Logistic Regression
            train_logistic_regression()
            result_message = "Mô hình đã được huấn luyện và lưu thành công."
        except Exception as e:
            result_message = f"Có lỗi xảy ra trong quá trình huấn luyện: {str(e)}"
        return self.render('admin/model.html', result_message=result_message)

    @expose('/predict', methods=['POST'])
    def predict(self):
        try:
            quantity_book = request.form.get('quantity_book', type=int)
            penalty = request.form.get('penalty', type=int)
            days_late = request.form.get('days_late', type=int)

            # Kiểm tra nếu giá trị là None
            if quantity_book is None or penalty is None or days_late is None:
                result_message = "Vui lòng nhập tất cả các trường thông tin."
            else:
                input_data = np.array([[quantity_book, penalty, days_late]])

                model = load_model()
                if model is None:
                    result_message = "Mô hình không được tải. Vui lòng kiểm tra file 'logistic_model.pkl'."
                else:
                    prediction = model.predict(input_data)
                    result_message = "Trả đúng hạn" if prediction[0] == 1 else "Trả trễ hạn"
        except Exception as e:
            result_message = f"Có lỗi xảy ra: {str(e)}"

        return render_template('admin/model.html', result_message=result_message)

def train_logistic_regression():
    data = Borrowing_Receipt.query.all()
    records = []
    for receipt in data:
        returned_on_time = 1 if receipt.return_date and receipt.return_date <= receipt.due_date else 0
        records.append({
            'quantity_book': receipt.quantity_book,
            'penalty': receipt.penalty,
            'days_late': (receipt.return_date - receipt.due_date).days if receipt.return_date else 0,
            'returned_on_time': returned_on_time
        })

    df = pd.DataFrame(records)
    X = df[['quantity_book', 'penalty', 'days_late']]
    y = df['returned_on_time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Đường dẫn thư mục và file mô hình
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    model_path = os.path.join(model_dir, 'logistic_model.pkl')

    # Lưu mô hình
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Kiểm tra xem file có được tạo ra không
    if os.path.exists(model_path):
        print("Mô hình đã được lưu thành công.")
    else:
        print("Lỗi: Mô hình không được lưu.")
def load_model():
    # Đường dẫn thư mục và file mô hình
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_path = os.path.join(model_dir, 'logistic_model.pkl')

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("Lỗi: file 'logistic_model.pkl' không tìm thấy.")
        return None
    except pickle.PickleError as e:
        print(f"Lỗi pickle: {e}")
        return None
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None


admin.add_view(ProductView(Book, db.session))
admin.add_view(UserView(User, db.session))
admin.add_view(ReviewModel(Review, db.session))
admin.add_view(BorrowingModel(Borrowing_Receipt, db.session))
admin.add_view(ChartView(name='Book Statistics', endpoint='book_stats'))
admin.add_view((TotalAmount(name='Total Amount')))
admin.add_view(ForecastView(name='Forecast', endpoint='forecast'))
admin.add_view(BookAnalysisView(name='Book Analysis', endpoint='book_analysis', category='Reports'))
admin.add_view(ModelView(name='Model Management', endpoint='model'))


