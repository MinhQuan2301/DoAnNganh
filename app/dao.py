# from app import db
# from app.models import Borrowing_Receipt, Book
# from sqlalchemy import func
#
#
# def get_annual_borrowing_data(self, year=2023):
#     # Truy vấn lấy tất cả các sách đã mượn trong năm 2023
#     data = db.session.query(
#         Borrowing_Receipt.book_title,
#         func.sum(Borrowing_Receipt.quantity_book).label('total_borrowed')
#     ).filter(
#         func.extract('year', Borrowing_Receipt.borrow_date) == year
#     ).group_by(
#         Borrowing_Receipt.book_title
#     ).all()
#
#     # Trả về danh sách dữ liệu mượn
#     return data
#
#
# def aggregate_annual_borrowing(self, borrowing_data):
#     aggregated_data = {}
#
#     for book_title, total_borrowed in borrowing_data:
#         if book_title not in aggregated_data:
#             aggregated_data[book_title] = 0
#         aggregated_data[book_title] += total_borrowed
#
#     return aggregated_data
#
#
# def get_all_books(self):
#     # Truy vấn tất cả sách từ bảng Book
#     books = db.session.query(Book.title, Book.quantity).all()
#     return books
#
#
# def analyze_books(self, all_books, aggregated_data):
#     books_to_remove = []
#     books_to_add = []
#
#     for book in all_books:
#         book_title = book.title
#         book_quantity = book.quantity
#         total_borrowed = aggregated_data.get(book_title, 0)
#
#         # Xác định tỉ lệ mượn sách
#         borrow_ratio = total_borrowed / book_quantity if book_quantity > 0 else 0
#
#         # Đưa ra quyết định dựa trên tỉ lệ mượn
#         if total_borrowed == 0:  # Sách không được mượn lần nào
#             books_to_remove.append((book_title, total_borrowed))
#         elif borrow_ratio < 0.1:  # Tỉ lệ mượn thấp hơn 10% so với số lượng
#             books_to_remove.append((book_title, total_borrowed))
#         elif borrow_ratio > 0.5:  # Tỉ lệ mượn cao hơn 50% so với số lượng
#             books_to_add.append((book_title, total_borrowed))
#
#     return books_to_remove, books_to_add
