from flask import redirect, request, render_template_string
from sklearn.linear_model import LogisticRegression

from admin import *
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64


@app.route("/")
def home():
    return redirect('/admin')


@app.route('/admin/chartstatsview', methods=['GET'])
def search_suggestions():
    books = Book.query.all()
    data = pd.DataFrame([(book.title, book.book_category, book.star_rating) for book in books],
                        columns=['title', 'book_category', 'star_rating'])
    keyword = request.args.get('kwd', '').strip().lower()
    relevant_book = data[data['book_category'].str.lower() == keyword]
    if relevant_book.empty:
        return render_template_string("<p>Không tìm thấy giá trị nào.</p>")
    relevant_book = relevant_book.sort_values('star_rating', ascending=False)

    # Tăng kích thước hình ảnh
    plt.figure(figsize=(12, len(relevant_book) * 0.8))
    plt.barh(relevant_book['title'], relevant_book['star_rating'], color='skyblue')
    plt.ylabel('Title')
    plt.xlabel('Star Rating')
    plt.title(f'Books in Category "{keyword}"')

    # Sử dụng tight_layout và bbox_inches để điều chỉnh hình ảnh
    plt.tight_layout(pad=3.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return render_template_string(
        '''
        <h2>Kết quả tìm kiếm cho "{{ keyword }}"</h2>
        <img src="data:image/png;base64,{{ image }}" style="display: block; margin: auto; max-width: 100%; height: auto;"/>
        ''', image=image, keyword=keyword
    )


if __name__ == "__main__":
    app.run(debug=True)
