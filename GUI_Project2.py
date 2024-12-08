import streamlit as st
import pandas as pd
import pickle
import joblib
import re
from underthesea import word_tokenize, pos_tag, sent_tokenize

# function cần thiết Project 2:
def get_recommendations(df, ma_san_pham, cosine_sim, nums=5):
    # Get the index of the product that matches the ma_san_pham
    matching_indices = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
    if not matching_indices:
        print(f"No product found with ID: {ma_san_pham}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the nums most similar products (Ignoring the product itself)
    sim_scores = sim_scores[1:nums+1]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top n most similar products as a DataFrame
    return df.iloc[product_indices]

# Hiển thị đề xuất ra bảng
def display_recommended_products(recommended_products, cols=5):
    for i in range(0, len(recommended_products), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:   
                    st.write(product['ten_san_pham'])                    
                    expander = st.expander(f"Mô tả")
                    product_description = product['mo_ta']
                    truncated_description = ' '.join(product_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           


# FUNTION cần thiết project1:
#  TIỀN XỬ LÝ DỮ LIỆU
def chuyen_chu_thuong(text):
	return text.lower()

def remove_html(txt):
    if pd.isnull(txt):  # Handle NaN values
        return txt
    return re.sub(r'<[^>]*>', '', txt)



def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = re.sub(r'\s+', ' ', document).strip()
    return document

# Load file dữ liệu
#LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()


def process_text(text, emoji_dict, teen_dict, wrong_lst):
    # Ensure the input is a string and convert to lowercase
    text = str(text)
    document = text.lower()
    document = document.replace("’", '')
    document = re.sub(r'\.+', ".", document)  # Normalize consecutive periods to a single period
    new_sentence = ''

    # Process each sentence in the text
    for sentence in sent_tokenize(document):
        # Slang normalization using teen_dict
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())

        # Remove undesired words from the sentence
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())

        # Append processed sentence to the new document
        new_sentence = new_sentence + sentence + '. '

    # Normalize spaces and strip leading/trailing whitespace
    document = re.sub(r'\s+', ' ', new_sentence).strip()
    return document




def process_noi_dung_binh_luan(text, emoji_dict, teen_dict, wrong_lst):
    try:
        print("Original:", text)
        # Step 1: Convert to lowercase
        text = chuyen_chu_thuong(text)
        # Step 2: preprocessing
        text = process_text(text, emoji_dict, teen_dict, wrong_lst)

    except Exception as e:
        print(f"Error processing text: {e}")
        return text  # Return original text on error
    return text


# Load positive and negative words from files:
with open("positive_VN.txt", "r", encoding="utf-8") as file:
    positive_words = file.read().splitlines()

with open("negative_VN.txt", "r", encoding="utf-8") as file:
    negative_words = file.read().splitlines()

# Load positive and negative emojis from files
with open("positive_emojis.txt", "r", encoding="utf-8") as file:
    positive_emojis = file.read().splitlines()

with open("negative_emoji.txt", "r", encoding="utf-8") as file:
    negative_emojis = file.read().splitlines()

# Define a function to count occurrences of words and emojis
def count_occurrences(text, items):
    return sum(text.count(item) for item in items)


# Thiết kế menu
menu = ["Mô tả bài toán", "Phân tích bình luận", "Giới thiệu sản phẩm"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Lã Đình Điền-Nguyễn Quỳnh Giang""")
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                 Khuất Thị Thúy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 14/12/2024""")

if choice == 'Mô tả bài toán':    
    st.title("Mô tả bài toán HASAKI")
    #st.subheader("Mô tả bài toán")
    st.write(""" #### HASAKI.VN - MỸ PHẨM VÀ CHĂM SÓC SẮC ĐẸP""")  
    st.write(""" ###### Khách hàng có thể lên đây để lựa chọn sản phẩm, xem các đánh giá/nhận xét, mua sản phẩm""")
    st.write(""" ###### Hệ thống cửa hàng trải dài trên toàn quốc; và hiện đang là đối tác phân phối chiến lược tại thị trường Việt Nam của hàng loạt thương hiệu lớn""")
    st.write(""" #### BÀI TOÁN 1: PHÂN TÍCH BÌNH LUẬN CỦA KHÁCH HÀNG VỀ SẢN PHẨM""")  
    st.write(""" ###### Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhãn hàng hiểu khách hàng rõ hơn, biết họ đánh giá gì về sản phẩm, từ đó có thể cải thiện chất lượng sản phẩm cũng như các dịch vụ đi kèm""")
    st.image("Bai_toan1.jpg")
    st.write(""" #### BÀI TOÁN 2: GỢI Ý SẢN PHẨM CHO KHÁCH HÀNG""")  
    st.write(""" ###### Xây dựng Recommender System giúp đề xuất sản phẩm phù hợp tới người dùng cho người dùng""")
    st.image("Bai_toan2.jpg")

elif choice == 'Giới thiệu sản phẩm':
    # Hiển thị tiêu đề
    st.title("Recomender System for Hasaki")
    st.write("Chào mừng bạn đến với hệ thống hỗ trợ của Hasaki")
    st.image('Banner_Hasaki.png', width=200)
    st.write('###                      THÔNG TIN SẢN PHẨM:')

    # Đọc dữ liệu đánh giá
    df_danh_gia = pd.read_csv('Danh_gia.csv')
    df_san_pham = pd.read_csv('San_pham.csv')
    df_khach_hang = pd.read_csv('Khach_hang.csv')
    df_tong_hop = pd.merge(df_danh_gia, df_san_pham, on='ma_san_pham')
    # Lấy 50 sản phẩm
    random_tong_hop = df_tong_hop.head(n=50)
    #print(random_danh_gia)

    st.session_state.random_tong_hop = random_tong_hop

    # Open and read file to cosine_sim_new
    with open('cosine_similarity_matrix.pkl', 'rb') as f:
        cosine_sim_new = joblib.load(f)

    ###### Giao diện Streamlit ######
    #st.image('hasaki_banner.jpg', use_column_width=True)

    # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
    if 'selected_ma_san_pham' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_ma_san_pham = None

    # Theo cách cho người dùng chọn sản phẩm từ dropdown
    # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in st.session_state.random_tong_hop.iterrows()]
    st.session_state.random_tong_hop
    # Tạo một dropdown với options là các tuple này
    selected_product = st.selectbox(
        "Chọn sản phẩm",
        options=product_options,
        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
    )
    # Display the selected product
    st.write("Bạn đã chọn:", selected_product)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_ma_san_pham = selected_product[1]

    if st.session_state.selected_ma_san_pham:
        st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
        # Hiển thị thông tin sản phẩm được chọn
        selected_product = df_tong_hop[df_tong_hop['ma_san_pham'] == st.session_state.selected_ma_san_pham]

        if not selected_product.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_product['ten_san_pham'].values[0])

            product_description = selected_product['mo_ta'].values[0]
            truncated_description = ' '.join(product_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Các sản phẩm liên quan:')
            recommendations = get_recommendations(df_tong_hop, st.session_state.selected_ma_san_pham, cosine_sim=cosine_sim_new, nums=3) 
            display_recommended_products(recommendations, cols=3)
        else:
            st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")
else:
    # Hiển thị tiêu đề
    st.title("Sentiment Analysis for Hasaki")
    st.write("Chào mừng bạn đến với hệ thống đánh giá của Hasaki")
    st.image('Banner_Hasaki.png', width=200)
    st.write('###                      THÔNG TIN SẢN PHẨM:')
    # Đọc dữ liệu đánh giá
    df_danh_gia = pd.read_csv('Danh_gia.csv')
    df_san_pham = pd.read_csv('San_pham.csv')
    df_khach_hang = pd.read_csv('Khach_hang.csv')
    df_tong_hop = pd.merge(df_danh_gia, df_san_pham, on='ma_san_pham')
    # Lấy 50 sản phẩm
    random_danh_gia = df_tong_hop.head(n=50)
    #print(random_danh_gia)

    st.session_state.random_danh_gia = random_danh_gia


    # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
    if 'selected_ma_san_pham' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_ma_san_pham = None

    # Theo cách cho người dùng chọn sản phẩm từ dropdown
    # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID


    danh_gia_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in st.session_state.random_danh_gia.iterrows()]
    st.session_state.random_danh_gia
    # Tạo một dropdown với options là các tuple này
    st.write('###                      LỰA CHỌN SẢN PHẨM BÌNH LUẬN:')
    selected_danh_gia = st.selectbox(
        "",
        options=danh_gia_options,
        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
    )
    # Display the selected product
    st.write("Bạn đã chọn:", selected_danh_gia)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_ma_san_pham = selected_danh_gia[1]

    if st.session_state.selected_ma_san_pham:
        st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
        # Hiển thị thông tin sản phẩm được chọn
        selected_product = df_tong_hop[df_tong_hop['ma_san_pham'] == st.session_state.selected_ma_san_pham]

        if not selected_product.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_product['ten_san_pham'].values[0])

            product_description = selected_product['mo_ta'].values[0]
            truncated_description = ' '.join(product_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

        else:
            st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")

    # Đọc mô hình đã lưu
    model = joblib.load('DecisionTreeClassifier_AdaBoost_model.pkl')

    # Đề nghị người dùng lựa chọn và nhận xét về sản phẩm này.
    st.write('###                      BẠN HÃY BÌNH LUẬN VỀ SẢN PHẨM ĐÃ CHỌN:')
    binh_luan = st.text_input("", "viết bình luận của bạn....")
    st.write("Bình luận:", binh_luan)
    binh_luan = process_noi_dung_binh_luan(binh_luan, emoji_dict, teen_dict, wrong_lst)

    # # Dự đoán với mô hình đã tải
    if binh_luan != "viết bình luận của bạn....":
        so_tu_tich_cuc = count_occurrences(binh_luan, positive_words) if isinstance(binh_luan, str) else 0
        so_tu_tieu_cuc = count_occurrences(binh_luan, negative_words) if isinstance(binh_luan, str) else 0
        so_emoji_tich_cuc = count_occurrences(binh_luan, positive_emojis) if isinstance(binh_luan, str) else 0
        so_emoji_tieu_cuc = count_occurrences(binh_luan, negative_emojis) if isinstance(binh_luan, str) else 0

        X_val = [[so_tu_tich_cuc, so_tu_tieu_cuc, so_emoji_tich_cuc, so_emoji_tieu_cuc]]

        predictions = model.predict(X_val)
        if int(predictions) == 0:
            danh_gia = "TIÊU CỰC"
        elif int(predictions) == 1:
            danh_gia = "TÍCH CỰC"
        else:
            danh_gia = "TRUNG TÍNH"

        st.write(f"Bình luận về sản phẩm {selected_danh_gia[0]} của bạn là:", danh_gia)

