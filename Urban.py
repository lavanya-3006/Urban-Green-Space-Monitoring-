import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sample Tamil Nadu district data (Replace with actual values)
district_data = {
    "Chennai": 22.5,  # Green Space %
    "Coimbatore": 38.2,
    "Madurai": 31.4,
    "Tiruchirappalli": 29.8,
    "Salem": 27.3,
    "Erode": 35.6,
    "Vellore": 28.0,
    "Tirunelveli": 40.1,
    "Kanyakumari": 43.7,  # Highest Green Space %
    "Namakkal": 33.9  # User's location
}

def extract_green_space(image):
    """Detect green areas in an urban image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 40, 40])  
    upper_green = np.array([85, 255, 255])  

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    
    total_pixels = image.shape[0] * image.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100
    
    return green_percentage, mask

def estimate_oxygen(green_percentage, area_km2=1.0):
    """Estimate oxygen production based on green space percentage."""
    oxygen_per_tree_kg = 118  
    trees_per_km2 = 250000  

    tree_coverage_factor = max(green_percentage / 100 * 0.5, 0.01)
    estimated_trees = max(trees_per_km2 * area_km2 * tree_coverage_factor, 1)  
    oxygen_output_kg = estimated_trees * oxygen_per_tree_kg

    people_supported = max(oxygen_output_kg / 180, 1)  

    return oxygen_output_kg, people_supported

# Convert district data to DataFrame
df = pd.DataFrame(list(district_data.items()), columns=["District", "Green Space %"])
df["Oxygen (kg/year)"] = df["Green Space %"].apply(lambda x: estimate_oxygen(x)[0])
df["People Supported"] = df["Green Space %"].apply(lambda x: estimate_oxygen(x)[1])

# Identify Tamil Nadu's highest & lowest oxygen-producing districts
highest_district = df.loc[df["Oxygen (kg/year)"].idxmax()]
lowest_district = df.loc[df["Oxygen (kg/year)"].idxmin()]

# Streamlit App
st.title("🌴 Urban Green Space & Tamil Nadu Oxygen Supply Comparison 🌲")
st.write("Upload an image to estimate oxygen production and compare it with Tamil Nadu's most & least oxygen-generating districts.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    green_coverage, mask = extract_green_space(image)
    oxygen_output, people_supported = estimate_oxygen(green_coverage)

    st.write(f"**Your Image's Green Space Coverage:** {green_coverage:.2f}%")
    st.write(f"**Estimated Oxygen Production:** {oxygen_output:.2f} kg/year")
    st.write(f"**Estimated People Supported:** {people_supported:.2f} per year")

    # Tamil Nadu Comparisons
    st.subheader("District-wise Oxygen Production Comparison")
    st.dataframe(df)

    st.subheader(f"Highest Oxygen-Producing District: {highest_district['District']}")
    st.write(f"**Green Space:** {highest_district['Green Space %']:.2f}%")
    st.write(f"**Oxygen Output:** {highest_district['Oxygen (kg/year)']:.2f} kg/year")
    st.write(f"**People Supported:** {highest_district['People Supported']:.2f} per year")

    st.subheader(f"Lowest Oxygen-Producing District: {lowest_district['District']}")
    st.write(f"**Green Space:** {lowest_district['Green Space %']:.2f}%")
    st.write(f"**Oxygen Output:** {lowest_district['Oxygen (kg/year)']:.2f} kg/year")
    st.write(f"**People Supported:** {lowest_district['People Supported']:.2f} per year")

    st.image(mask, caption="Detected Green Areas", use_container_width=True)

    # Generate Graph: Comparison of User Image vs Districts
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = ["Your Image", highest_district["District"], lowest_district["District"]]
    values = [oxygen_output, highest_district["Oxygen (kg/year)"], lowest_district["Oxygen (kg/year)"]]

    ax.bar(categories, values, color=['orange', 'green', 'red'])
    ax.set_ylabel("Oxygen Production (kg/year)")
    ax.set_title("Comparison of Oxygen Output")
    
    st.pyplot(fig)

st.write("Extend this with real GIS data or satellite imagery to enhance accuracy!")