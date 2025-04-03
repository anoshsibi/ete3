import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import random
from faker import Faker
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources python -m streamlit run etepy.py
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Hackathon Event Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .st-bw {
        background-color: #ffffff;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Create directory for static assets if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Define the function to generate data if it doesn't exist
def generate_hackathon_data():
    """Generate synthetic hackathon data"""
    # Check if data already exists
    if os.path.exists("hackathon_data.csv"):
        return pd.read_csv("hackathon_data.csv")
    
    # If not, generate new data
    # (Copying the data generation code from the dataset generation script)
    np.random.seed(42)
    random.seed(42)
    fake = Faker()
    Faker.seed(42)

    NUM_PARTICIPANTS = 350
    HACKATHON_DOMAINS = ['Web Development', 'VR Game Development', 'Blockchain', 'AR App Development', 'Mobile App']
    STATES = [ 'Karnataka', 'Tamil Nadu', 'Delhi', 'Goa', 'Kerala']
    COLLEGES = ['Christ University', 'St.Josephs College', 'Pes University', 'Rv University', 'KJC']

    start_date = datetime(2025, 4, 1)

    def generate_feedback(domain):
        domain_feedback = {
            'Web Development': [
                "The frontend workshops were very helpful.",
                "Needed more time for UI/UX design phase.",
                "Great opportunity to showcase my responsive design skills.",
                "The mentors that organised the event were amazing!",
                "Would have liked more focus on backend technologies.",
                "Thoroughly enjoyed working with APIs.",
                "The CSS challenges were fun but challenging.",
                "Learned a lot about modern JavaScript frameworks.",
                "The deployment session saved us a lot of time.",
                "Networking with industry experts was valuable."
            ],
            'VR Game Development': [
            "The datasets provided were comprehensive.",
            "Would have liked more time for model training.",
            "The computing resources were sufficient.",
            "Great introductory workshops on neural networks.",
            "Challenging problem statements pushed our limits.",
            "The feature engineering workshop was eye-opening.",
            "Could have used more cloud computing credits.",
            "Learning about ethical AI was enlightening.",
            "The NLP challenges were my favorite part.",
            "Great balance between theory and application."
            ],
            'Blockchain': [
            "Smart contract development was the highlight.",
            "More resources on Solidity would have been helpful.",
            "The blockchain security workshop was eye-opening.",
            "Great introduction to Web3 development.",
            "Enjoyed building decentralized applications.",
            "The crypto economics session was complex but useful.",
            "Would have liked more focus on NFT development.",
            "The mentors were knowledgeable about DeFi.",
            "Hands-on experience with Ethereum was valuable.",
            "Learning about consensus algorithms was challenging."
            ],
            'AR App Development': [
            "Hardware availability was excellent.",
            "More time for sensor calibration would have helped.",
            "The Arduino workshops were practical and useful.",
            "Great experience connecting physical and digital worlds.",
            "The IoT security session was eye-opening.",
            "Would have liked more advanced sensor options.",
            "Enjoyed working with Raspberry Pi.",
            "The networking protocols workshop was detailed.",
            "Real-world problem statements made it exciting.",
            "More time for prototyping would have been better."
            ],
            'Mobile App': [
                "Learned a lot about cross-platform development.",
                "The UI/UX workshops were very practical.",
                "Great balance between Android and iOS development.",
                "The Flutter session was the highlight for me.",
                "Would have liked more focus on app store optimization.",
                "The API integration workshop saved us time.",
                "Challenging but rewarding experience overall.",
                "The mobile security session was very informative.",
                "Enjoyed working with React Native.",
                "Great mentorship for debugging native issues."
            ]
        }
        
        base_feedback = random.choice(domain_feedback[domain])
        
        sentiment = random.choice(["Overall, I enjoyed the event! ", "The event was good. ", 
                                "It was a great learning experience. ", ""])
        improvement = random.choice(["Could improve on time management. ", "Would love to participate again. ", 
                                    "Looking forward to next year! ", ""])
        
        return sentiment + base_feedback + " " + improvement

    data = []
    participant_id = 1000
    
    for _ in range(NUM_PARTICIPANTS):
        participant_id += 1
        domain = random.choice(HACKATHON_DOMAINS)
        day = random.randint(1, 3)
        
        if random.random() > 0.2 * day:
            state = random.choice(STATES)
            college = random.choice(COLLEGES)
            
            if college in ['Christ University', 'St.Josephs College', 'Pes University', 'Rv University', 'KJC'] and random.random() < 0.8:
                state = 'Karnataka'
            elif college in ['Maharajas', 'Kerala University', 'Kerala Technological University'] and random.random() < 0.8:
                state = 'Kerala'
            elif college in ['Goa IIT', 'Goa University'] and random.random() < 0.8:
                state = 'Goa'
            
            team_size = random.randint(1, 4)
            
            if domain in ['Mobile App', 'VR Game Development', 'Blockchain', 'AR App Development']:
                team_size = min(5, max(2, int(np.random.normal(3, 0.5))))
            elif domain == 'Web Development':
                team_size = min(4, max(2, int(np.random.normal(3.2, 0.4))))
            
            completion_status = random.choices(
                ['Completed', 'Partial', 'Not Completed'],
                weights=[0.75, 0.2, 0.05]
            )[0]
            
            if completion_status == 'Completed':
                satisfaction = max(3, min(10, int(np.random.normal(8, 1))))
            elif completion_status == 'Partial':
                satisfaction = max(3, min(10, int(np.random.normal(6, 1))))
            else:
                satisfaction = max(1, min(10, int(np.random.normal(4, 1))))
            
            feedback = generate_feedback(domain)
            
            data.append({
                'ParticipantID': f"P{participant_id}",
                'Day': day,
                'RegistrationDate': (start_date + pd.Timedelta(days=day-1)).strftime('%Y-%m-%d'),
                'Domain': domain,
                'State': state,
                'College': college,
                'TeamSize': team_size,
                'CompletionStatus': completion_status,
                'SatisfactionScore': satisfaction,
                'Feedback': feedback
            })
    
    df = pd.DataFrame(data)
    df.to_csv("hackathon_data.csv", index=False)
    return df

# Generate or load sample images for the domains
def generate_domain_images():
    """Generate sample images for each domain if they don't exist"""
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Create domain folders and copy images
    domains = ['Web Development', 'VR Game Development', 'Blockchain', 'AR App Development', 'Mobile App']
    source_dir = r"C:\Users\Anosh Sibi\Downloads\ete1"
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    for domain in domains:
        domain_folder = f"static/{domain.replace(' ', '_')}"
        if not os.path.exists(domain_folder):
            os.makedirs(domain_folder)
        
        # Copy images for each day
        for day in range(1, 4):
            target_path = f"{domain_folder}/day_{day}.png"
            if not os.path.exists(target_path):
                # Use modulo to cycle through available images
                source_image = image_files[(day + domains.index(domain)) % len(image_files)]
                source_path = os.path.join(source_dir, source_image)
                
                # Open and save the image
                try:
                    img = Image.open(source_path)
                    # Resize image to a reasonable size
                    img.thumbnail((800, 600))
                    img.save(target_path)
                except Exception as e:
                    st.error(f"Error processing image {source_image}: {str(e)}")

# Function to create a wordcloud from domain-specific feedback
def create_wordcloud(df, domain=None):
    """Create a wordcloud from feedback"""
    if domain:
        text = " ".join(df[df['Domain'] == domain]['Feedback'].tolist())
    else:
        text = " ".join(df['Feedback'].tolist())
    
    # Process text
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    
    # Generate wordcloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          max_words=100, contour_width=3, contour_color='steelblue').generate(" ".join(filtered_text))
    
    # Create and return the figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def display_all_images():
    """Display all images in a grid layout"""
    source_dir = r"C:\Users\Anosh Sibi\Downloads\ete1"
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # Create columns for the grid (3 images per row)
    cols = st.columns(3)
    
    for idx, img_file in enumerate(image_files):
        with cols[idx % 3]:
            img_path = os.path.join(source_dir, img_file)
            try:
                img = Image.open(img_path)
                st.image(img, caption=img_file, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image {img_file}: {str(e)}")

def image_gallery(day=None):
    """Create image gallery based on day filter"""
    st.header("📸 Image Gallery")
    
    # Display all images first
    st.subheader("All Images")
    display_all_images()
    
    # Add filters
    st.subheader("Filtered Gallery")
    col1, col2 = st.columns(2)
    with col1:
        selected_domain = st.selectbox("Select Domain", 
            ['All'] + ['Web Development', 'VR Game Development', 'Blockchain', 'AR App Development', 'Mobile App'])
    with col2:
        selected_day = st.selectbox("Select Day", ['All'] + list(range(1, 4)))
    
    # Show filtered images
    if selected_domain != 'All' or selected_day != 'All':
        st.write("Filtered Images:")
        domains = [selected_domain] if selected_domain != 'All' else ['Web Development', 'VR Game Development', 'Blockchain', 'AR App Development', 'Mobile App']
        cols = st.columns(len(domains))
        
        for i, domain in enumerate(domains):
            domain_folder = f"static/{domain.replace(' ', '_')}"
            if not os.path.exists(domain_folder):
                continue
                
            with cols[i]:
                st.write(f"**{domain}**")
                if selected_day != 'All':
                    img_path = f"{domain_folder}/day_{selected_day}.png"
                    if os.path.exists(img_path):
                        cols[i].image(img_path, caption=f"{domain} - Day {selected_day}", use_container_width=True)
                else:
                    # Show random day image if no day specified
                    day_sample = random.randint(1, 3)
                    img_path = f"{domain_folder}/day_{day_sample}.png"
                    if os.path.exists(img_path):
                        cols[i].image(img_path, caption=f"{domain} - Day {day_sample}", use_container_width=True)
    
    # Image Processing Section
    st.subheader("Image Processing")
    uploaded_file = st.file_uploader("Choose an image to process", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            # Open and validate the image
            image = Image.open(uploaded_file)
            image = image.convert('RGB')  # Convert to RGB to ensure compatibility
            
            # Image filter options
            filter_option = st.selectbox(
                "Select Filter",
                ["Original", "Grayscale", "Sepia", "Invert", "Enhance Contrast", "Enhance Brightness"]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                if filter_option != "Original":
                    st.write(f"Filtered Image ({filter_option})")
                    try:
                        processed_image = apply_image_filters(image, filter_option)
                        st.image(processed_image, use_container_width=True)
                    except Exception as filter_error:
                        st.error(f"Error applying filter: {str(filter_error)}")
                        st.image(image, use_container_width=True)
                else:
                    st.write("Original Image (No Filter)")
                    st.image(image, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.warning("Please make sure you've uploaded a valid image file.")

def apply_image_filters(image, filter_name):
    """Apply image filters to the selected image"""
    img = image.copy()
    
    if filter_name == "Grayscale":
        img = img.convert("L").convert("RGB")
    elif filter_name == "Sepia":
        arr = np.array(img)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        new_r = (0.393 * r + 0.769 * g + 0.189 * b).astype(np.uint8)
        new_g = (0.349 * r + 0.686 * g + 0.168 * b).astype(np.uint8)
        new_b = (0.272 * r + 0.534 * g + 0.131 * b).astype(np.uint8)
        new_rgb = np.stack([new_r, new_g, new_b], axis=2)
        img = Image.fromarray(new_rgb)
    elif filter_name == "Invert":
        arr = np.array(img)
        arr = 255 - arr
        img = Image.fromarray(arr)
    elif filter_name == "Enhance Contrast":
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
    elif filter_name == "Enhance Brightness":
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.3)
    
    return img

# Main application
def main():
    # Generate data and images
    df = generate_hackathon_data()
    generate_domain_images()
    
    # Sidebar - Title and Filters
    st.sidebar.title("🚀 Hackathon Dashboard")
    st.sidebar.subheader("Filters")
    
    # Domain filter
    domain_filter = st.sidebar.multiselect(
        "Select Domains",
        options=df['Domain'].unique(),
        default=df['Domain'].unique()
    )
    
    # State filter
    state_filter = st.sidebar.multiselect(
        "Select States",
        options=df['State'].unique(),
        default=df['State'].unique()[:3]  # Default to first 3 states
    )
    
    # College filter
    college_filter = st.sidebar.multiselect(
        "Select Colleges",
        options=df['College'].unique(),
        default=df['College'].unique()[:3]  # Default to first 5 colleges
    )
    
    # Day filter
    day_filter = st.sidebar.multiselect(
        "Select Days",
        options=df['Day'].unique(),
        default=df['Day'].unique()
    )
    
    # Apply filters
    filtered_df = df[
        df['Domain'].isin(domain_filter) &
        df['State'].isin(state_filter) &
        df['College'].isin(college_filter) &
        df['Day'].isin(day_filter)
    ]
    
    # Main content area with tabs
    st.title("🚀 Hackathon Event Analysis Dashboard")
    
    tabs = st.tabs(["📊 Dashboard", "💬 Feedback Analysis", "🖼️ Image Gallery"])
    
    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("Hackathon Participation Overview")
        
        # Display key metrics in cards
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        metric_col1.metric(
            "Total Participants", 
            len(filtered_df),
            f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None
        )
        
        avg_satisfaction = round(filtered_df['SatisfactionScore'].mean(), 2)
        metric_col2.metric(
            "Avg. Satisfaction", 
            avg_satisfaction,
            round(avg_satisfaction - df['SatisfactionScore'].mean(), 2) if len(filtered_df) != len(df) else None
        )
        
        completion_rate = round(len(filtered_df[filtered_df['CompletionStatus'] == 'Completed']) / len(filtered_df) * 100, 2)
        metric_col3.metric(
            "Completion Rate (%)", 
            completion_rate,
            round(completion_rate - (len(df[df['CompletionStatus'] == 'Completed']) / len(df) * 100), 2) if len(filtered_df) != len(df) else None
        )
        
        avg_team_size = round(filtered_df['TeamSize'].mean(), 2)
        metric_col4.metric(
            "Avg. Team Size", 
            avg_team_size,
            round(avg_team_size - df['TeamSize'].mean(), 2) if len(filtered_df) != len(df) else None
        )
        
        st.markdown("---")
        
        # Domain-wise distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Domain-wise Participation")
            domain_counts = filtered_df['Domain'].value_counts().reset_index()
            domain_counts.columns = ['Domain', 'Count']
            
            fig_domain = px.bar(
                domain_counts, 
                x='Domain', 
                y='Count',
                color='Domain',
                text='Count',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_domain.update_layout(xaxis_title="Domain", yaxis_title="Number of Participants")
            st.plotly_chart(fig_domain, use_container_width=True)
        
        with col2:
            st.subheader("Day-wise Registration")
            day_counts = filtered_df['Day'].value_counts().reset_index()
            day_counts.columns = ['Day', 'Count']
            day_counts = day_counts.sort_values('Day')
            
            fig_day = px.line(
                day_counts, 
                x='Day', 
                y='Count',
                markers=True,
                text='Count',
                line_shape='spline',
                color_discrete_sequence=['#1f77b4']
            )
            fig_day.update_traces(textposition='top center')
            fig_day.update_layout(xaxis_title="Hackathon Day", yaxis_title="Number of Registrations")
            st.plotly_chart(fig_day, use_container_width=True)
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("State-wise Distribution")
            state_counts = filtered_df['State'].value_counts().reset_index()
            state_counts.columns = ['State', 'Count']
            
            fig_state = px.pie(
                state_counts, 
                values='Count', 
                names='State',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_state.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_state, use_container_width=True)
        
        with col4:
            st.subheader("College-wise Distribution")
            college_counts = filtered_df['College'].value_counts().nlargest(10).reset_index()
            college_counts.columns = ['College', 'Count']
            
            fig_college = px.bar(
                college_counts, 
                x='Count', 
                y='College',
                orientation='h',
                color='Count',
                color_continuous_scale='Viridis',
                text='Count'
            )
            fig_college.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_college, use_container_width=True)
        
        st.markdown("---")
        
        # Satisfaction scores by domain
        st.subheader("Satisfaction Score Analysis")
        
        col5, col6 = st.columns(2)
        
        with col5:
            domain_satisfaction = filtered_df.groupby('Domain')['SatisfactionScore'].mean().reset_index()
            domain_satisfaction.columns = ['Domain', 'Average Satisfaction']
            
            fig_satisfaction = px.bar(
                domain_satisfaction, 
                x='Domain', 
                y='Average Satisfaction',
                color='Domain',
                text=domain_satisfaction['Average Satisfaction'].round(2),
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_satisfaction.update_layout(
                xaxis_title="Domain", 
                yaxis_title="Average Satisfaction Score",
                yaxis=dict(range=[0, 10])
            )
            st.plotly_chart(fig_satisfaction, use_container_width=True)
        
        with col6:
            completion_status = filtered_df['CompletionStatus'].value_counts().reset_index()
            completion_status.columns = ['Status', 'Count']
            
            fig_completion = px.pie(
                completion_status, 
                values='Count', 
                names='Status',
                color='Status',
                color_discrete_map={
                    'Completed': '#2ecc71',
                    'Partial': '#f1c40f',
                    'Not Completed': '#e74c3c'
                }
            )
            fig_completion.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_completion, use_container_width=True)
    
    # Tab 2: Feedback Analysis
    with tabs[1]:
        st.subheader("Participant Feedback Analysis")
        
        # Domain selection for wordcloud
        domain_select = st.selectbox(
            "Select Domain for Feedback Analysis",
            options=['All'] + list(df['Domain'].unique())
        )
        
        # Create and display wordcloud
        if domain_select == 'All':
            wordcloud_fig = create_wordcloud(filtered_df)
            st.pyplot(wordcloud_fig)
        else:
            if len(filtered_df[filtered_df['Domain'] == domain_select]) > 0:
                wordcloud_fig = create_wordcloud(filtered_df, domain_select)
                st.pyplot(wordcloud_fig)
            else:
                st.warning(f"No data available for {domain_select} with current filters.")
        
        st.markdown("---")
        
        # Sentiment analysis by domain (simulated)
        st.subheader("Domain-wise Sentiment Analysis")
        
        # Create dummy sentiment scores based on satisfaction scores
        domain_sentiment = filtered_df.groupby('Domain').agg({
            'SatisfactionScore': ['mean', 'count']
        }).reset_index()
        domain_sentiment.columns = ['Domain', 'Average Satisfaction', 'Count']
        
        # Create pseudo-sentiment metrics
        domain_sentiment['Positive'] = domain_sentiment['Average Satisfaction'].apply(
            lambda x: round(x / 10 * 100, 1)
        )
        domain_sentiment['Neutral'] = domain_sentiment['Average Satisfaction'].apply(
            lambda x: round((10 - x) / 10 *50, 1)
        )
        domain_sentiment['Negative'] = 100 - domain_sentiment['Positive'] - domain_sentiment['Neutral']
        
        # Create stacked bar chart for sentiment analysis
        fig_sentiment = go.Figure()
        
        for sentiment, color in [('Positive', '#2ecc71'), ('Neutral', '#f1c40f'), ('Negative', '#e74c3c')]:
            fig_sentiment.add_trace(go.Bar(
                name=sentiment,
                x=domain_sentiment['Domain'],
                y=domain_sentiment[sentiment],
                marker_color=color
            ))
        
        fig_sentiment.update_layout(
            barmode='stack',
            title='Sentiment Distribution by Domain',
            xaxis_title='Domain',
            yaxis_title='Percentage',
            showlegend=True
        )
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
        # Display sample feedback
        st.subheader("Sample Feedback by Domain")
        selected_domain = st.selectbox(
            "Select Domain to View Feedback",
            options=filtered_df['Domain'].unique()
        )
        
        sample_feedback = filtered_df[filtered_df['Domain'] == selected_domain].sample(
            min(5, len(filtered_df[filtered_df['Domain'] == selected_domain]))
        )
        
        for _, row in sample_feedback.iterrows():
            st.write(f"🗣️ *\"{row['Feedback']}\"* - Satisfaction Score: {row['SatisfactionScore']}/10")
            # Tab 3: Image Gallery
    with tabs[2]:
        image_gallery()
        
        # Image Processing Section
        st.markdown("---")
        st.subheader("Image Processing")
        
        uploaded_file = st.file_uploader("Upload an image for processing", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read and display the original image
            image = Image.open(uploaded_file)
            
            # Image filter selection
            filter_option = st.selectbox(
                "Select Image Filter",
                ["Original", "Grayscale", "Sepia", "Invert", "Enhance Contrast", "Enhance Brightness"]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                if filter_option != "Original":
                    st.write(f"Processed Image ({filter_option})")
                    try:
                        processed_image = apply_image_filters(image, filter_option)
                        st.image(processed_image, use_container_width=True)
                    except Exception as filter_error:
                        st.error(f"Error applying filter: {str(filter_error)}")
                        st.image(image, use_container_width=True)
                else:
                    st.write("Original Image (No Filter)")
                    st.image(image, use_container_width=True)

if __name__ == "__main__":
    main()