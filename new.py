import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Load the dataset
df = pd.read_csv("Womens.csv")  # Replace "your_dataset.csv" with the path to your CSV file

# Function to resize the image
def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Function to convert the image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to crop the image
def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

# Function to rotate the image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Function to load the image
def load_image(image_path):
    return cv2.imread(image_path)

# Function to preprocess the text
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_tokens = [word for word in filtered_tokens if word not in string.punctuation and not re.match(r'^\W+$', word)]
    filtered_tokens = [word.lower() for word in filtered_tokens]
    clean_text = ' '.join(filtered_tokens)
    return clean_text

# Create the Streamlit app
def main():
    st.title("Product Reviews Dashboard")
    page = st.sidebar.selectbox("Choose a page", ["Overview", "Data Exploration", "Review Analysis", "3D Plot", "Image Processing", "Text Processing"])

    if page == "Overview":
        show_overview()
    elif page == "Data Exploration":
        show_data_exploration()
    elif page == "Review Analysis":
        show_review_analysis()
    elif page == "3D Plot":
        show_3d_plot()
    elif page == "Image Processing":
        show_image_processing()
    elif page == "Text Processing":
        show_text_processing()

# Define functions to show different pages
def show_overview():
    st.header("Dataset Overview")
    st.write("This dashboard provides an overview of the product reviews dataset.")

    st.subheader("Sample Data")
    st.write(df.head())

def show_data_exploration():
    st.header("Data Exploration")
    st.write("Explore the dataset.")

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    st.subheader("Distribution of Ratings")
    rating_counts = df['Rating'].value_counts()
    st.bar_chart(rating_counts)

def show_review_analysis():
    st.header("Review Analysis")
    st.write("Analyzing the reviews.")

    st.subheader("Average Rating")
    avg_rating = df['Rating'].mean()
    st.write(f"The average rating is: {avg_rating:.2f}")

    st.subheader("Reviews with Most Positive Feedback")
    top_reviews = df.nlargest(5, 'Positive Feedback Count')[['Review Text', 'Positive Feedback Count']]
    st.write(top_reviews)

def show_3d_plot():
    st.header("3D Plot")
    st.write("Visualizing the relationship between Age, Rating, and Positive Feedback Count.")

    fig = px.scatter_3d(df, x='Age', y='Rating', z='Positive Feedback Count')
    st.plotly_chart(fig)

def show_image_processing():
    st.header("Image Processing")
    st.write("Performing image processing tasks.")

    # Load the image
    image_path = "dress1.jpg"
    image = load_image(image_path)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_column_width=True)

    # Image processing options
    st.sidebar.subheader("Image Processing Options")
    resize_option = st.sidebar.checkbox("Resize Image")
    grayscale_option = st.sidebar.checkbox("Convert to Grayscale")
    crop_option = st.sidebar.checkbox("Crop Image")
    rotation_option = st.sidebar.checkbox("Rotate Image")

    # Perform image processing based on selected options
    processed_image = image.copy()

    if resize_option:
        new_width = st.sidebar.slider("New Width", 50, 1000, 300)
        new_height = st.sidebar.slider("New Height", 50, 1000, 300)
        processed_image = resize_image(processed_image, new_width, new_height)
        st.subheader("Resized Image")
        st.image(processed_image, caption="Resized Image", use_column_width=True)

    if grayscale_option:
        processed_image = convert_to_grayscale(processed_image)
        st.subheader("Grayscale Image")
        st.image(processed_image, caption="Grayscale Image", use_column_width=True, channels='GRAY')

    if crop_option:
        x = st.sidebar.slider("X", 0, processed_image.shape[1], 0)
        y = st.sidebar.slider("Y", 0, processed_image.shape[0], 0)
        width = st.sidebar.slider("Width", 1, processed_image.shape[1], processed_image.shape[1])
        height = st.sidebar.slider("Height", 1, processed_image.shape[0], processed_image.shape[0])
        processed_image = crop_image(processed_image, x, y, width, height)
        st.subheader("Cropped Image")
        st.image(processed_image, caption="Cropped Image", use_column_width=True)

    if rotation_option:
        angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
        processed_image = rotate_image(processed_image, angle)
        st.subheader("Rotated Image")
        st.image(processed_image, caption="Rotated Image", use_column_width=True)

def show_text_processing():
    st.header("Text Processing")
    st.write("Performing text processing tasks.")

    # Sample text
    text = """
    Absolutely wonderful - silky and sexy and comfortable
    Love this dress!  it's sooo pretty.  i happened to find it in a store, and i'm glad i did bc i never would have ordered it online bc it's petite.  i bought a petite and am 5'8".  i love the length on me- hits just a little below the knee.  would definitely be a true midi on someone who is truly petite.
    I had such high hopes for this dress and really wanted it to work for me. i initially ordered the petite small (my usual size) but i found this to be outrageously small. so small in fact that i could not zip it up! i reordered it in petite medium, which was just ok. overall, the top half was comfortable and fit nicely, but the bottom half had a very tight under layer and several somewhat cheap (net) over layers. imo, a major design flaw was the net over layer sewn directly into the zipper - it c
    I love, love, love this jumpsuit. it's fun, flirty, and fabulous! every time i wear it, i get nothing but great compliments!
    This shirt is very flattering to all due to the adjustable front tie. it is the perfect length to wear with leggings and it is sleeveless so it pairs well with any cardigan. love this shirt!!!
    I love tracy reese dresses, but this one is not for the very petite. i am just under 5 feet tall and usually wear a 0p in this brand. this dress was very pretty out of the package but its a lot of dress. the skirt is long and very full so it overwhelmed my small frame. not a stranger to alterations, shortening and narrowing the skirt would take away from the embellishment of the garment. i love the color and the idea of the style but it just did not work on me. i returned this dress.
    I aded this in my basket at hte last mintue to see what it would look like in person. (store pick up). i went with teh darkler color only because i am so pale :-) hte color is really gorgeous, and turns out it mathced everythiing i was trying on with it prefectly. it is a little baggy on me and hte xs is hte msallet size (bummer, no petite). i decided to jkeep it though, because as i said, it matvehd everything. my ejans, pants, and the 3 skirts i waas trying on (of which i ]kept all ) oops.
    I ordered this in carbon for store pick up, and had a ton of stuff (as always) to try on and used this top to pair (skirts and pants). everything went with it. the color is really nice charcoal with shimmer, and went well with pencil skirts, flare pants, etc. my only compaint is it is a bit big, sleeves are long and it doesn't go in petite. also a bit loose for me, but no xxs... so i kept it and wil ldecide later since the light color is already sold out in hte smallest size...
    I love this dress. i usually get an xs but it runs a little snug in bust so i ordered up a size. very flattering and feminine with the usual retailer flair for style.
    I'm 5"5' and 125 lbs. i ordered the s petite to make sure the length wasn't too long. i typically wear an xs regular in retailer dresses. if you're less busty (34b cup or smaller), a s petite will fit you perfectly (snug, but not tight). i love that i could dress it up for a party, or down for work. i love that the tulle is longer then the fabric underneath.
    Dress runs small esp where the zipper area runs. i ordered the sp which typically fits me and it was very tight! the material on the top looks and feels very cheap that even just pulling on it will cause it to rip the fabric. pretty disappointed as it was going to be my christmas dress this year! needless to say it will be going back.
    This dress is perfection! so pretty and flattering.
    More and more i find myself reliant on the reviews written by savvy shoppers before me and for the most past, they are right on in their estimation of the product. in the case of this dress-if it had not been for the reveiws-i doubt i would have even tried this. the dress is beautifully made, lined and reminiscent of the old retailer quality. it is lined in the solid periwinkle-colored fabric that matches the outer fabric print. tts and very form-fitting. falls just above the knee and does not rid
    "Bought the black xs to go under the larkspur midi dress because they didn't bother lining the skirt portion (grrrrrrrrrrr).
    my stats are 34a-28/29-36 and the xs fit very smoothly around the chest and was flowy around my lower half, so i would say it's running big.
    the straps are very pretty and it could easily be nightwear too.
    i'm 5'6"" and it came to just below my knees."
    This is a nice choice for holiday gatherings. i like that the length grazes the knee so it is conservative enough for office related gatherings. the size small fit me well - i am usually a size 2/4 with a small bust. in my opinion it runs small and those with larger busts will definitely have to size up (but then perhaps the waist will be too big). the problem with this dress is the quality. the fabrics are terrible. the delicate netting type fabric on the top layer of skirt got stuck in the zip
    I took these out of the package and wanted them to fit so badly, but i could tell before i put them on that they wouldn't. these are for an hour-glass figure. i am more straight up and down. the waist was way too small for my body shape and even if i sized up, i could tell they would still be tight in the waist and too roomy in the hips - for me. that said, they are really nice. sturdy, linen-like fabric, pretty color, well made. i hope they make someone very happy!
    Material and color is nice.  the leg opening is very large.  i am 5'1 (100#) and the length hits me right above my ankle.  with a leg opening the size of my waist and hem line above my ankle, and front pleats to make me fluffy, i think you can imagine that it is not a flattering look.  if you are at least average height or taller, this may look good on you.
    Took a chance on this blouse and so glad i did. i wasn't crazy about how the blouse is photographed on the model. i paired it whit white pants and it worked perfectly. crisp and clean is how i would describe it. launders well. fits great. drape is perfect. wear tucked in or out - can't go wrong.
    A flattering, super cozy coat.  will work well for cold, dry days and will look good with jeans or a dressier outfit.  i am 5' 5'', about 135 and the small fits great.
    I love the look and feel of this tulle dress. i was looking for something different, but not over the top for new year's eve. i'm small chested and the top of this dress is form fitting for a flattering look. once i steamed the tulle, it was perfect! i ordered an xsp. length was perfect too.
    "If this product was in petite, i would get the petite. the regular is a little long on me but a tailor can do a simple fix on that. 

    fits nicely! i'm 5'4, 130lb and pregnant so i bough t medium to grow into. """

    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    st.subheader("Preprocessed Text")
    st.write(preprocessed_text)

# Run the app
if __name__ == "__main__":
    main()
