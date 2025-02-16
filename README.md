# Real Estate Property Catalog

This web application allows users to view, filter, and explore real estate property listings uploaded as a CSV file. The application provides various filtering options such as price, surface area, operation type (e.g., sale or auction), and more. The properties are displayed in both a paginated list and on an interactive map.

## Features

- **CSV Upload**: Users can upload a CSV file containing real estate listings.
- **Language Support**: The app supports English and Greek.
- **Data Filtering**: Filter properties by price range, surface area, category, operation type, and more.
- **Property Display**: Each property is displayed with essential details like price, surface area, category, parking availability, and a brief description.
- **Pagination**: Property listings are paginated for easy browsing.
- **Interactive Map**: A map view with markers for each property location.
- **Detailed Property View**: Users can click on a property to view more details, including images, description, and link to the original listing.

## Requirements

- Python 3.7+
- Streamlit
- pandas
- folium
- streamlit-folium
- ast

## Installation

To set up and run the application, follow these steps:

### 1. Clone the Repository:
```bash
git clone https://github.com/Tsilkostas/Real_Estate_Application.git
cd Real_Estate_Application
```

### 2. Create and Activate a Virtual Environment:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install the required libraries using:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app:
```bash
streamlit run realEstate_app.py
```

### 5. Navigate to http://localhost:8501 to interact with the app.

## CSV File Format
The CSV file should have the following columns:

| Column Name         | Description |
|---------------------|-------------|
| `created_at`       | Date of property listing creation. |
| `updated_at`       | Date of last update. |
| `category_en`      | Category in English (e.g., Apartment, House, etc.). |
| `category_gr`      | Category in Greek (e.g., Διαμέρισμα, Κατοικία). |
| `category_source_en` | Source category in English. |
| `type_gr`         | Type of the property in Greek (e.g., Πωλητήριο, Ενοικίαση). |
| `operation`        | Operation type (e.g., Sale, Auction). |
| `lat`             | Latitude coordinate of the property. |
| `lng`             | Longitude coordinate of the property. |
| `surface`         | Surface area of the property in square meters. |
| `construction_year` | Year of construction. |
| `price`           | Price of the property in EUR. |
| `price_per_m2`    | Price per square meter. |
| `has_parking`     | Boolean indicating whether the property has parking. |
| `has_storage`     | Boolean indicating whether the property has storage. |
| `floor_num`       | Floor number of the property. |
| `floor_cnt`       | Total number of floors in the building. |
| `floor_min`       | Minimum floor number in the building. |
| `address_gr`      | Address of the property in Greek. |
| `description_gr`  | Description of the property in Greek. |
| `url`            | URL to the original listing. |
| `img_url`        | Image URLs (should be a list of URLs in string format). |
| `postcode`       | Postal code of the property. |

---


## How It Works

### 1. Upload a CSV File  
Click on the upload button and select your CSV file containing the property listings.

### 2. Select Filters  
Use the sidebar to filter the properties by:
- **Operation Type** (Sale, Auction, etc.)
- **Price Range**
- **Surface Area**
- **Category**
- **Construction Year**
- **Parking and Storage Availability**

### 3. View Property Listings  
The filtered properties are displayed in a list and paginated for easy navigation.

### 4. Map View  
The properties are also displayed on an interactive map, with markers for each location.

### 5. Property Details  
Click on a property card to view detailed information, including images, descriptions, and a link to the original listing.

---

## Localization

The application supports two languages:
- **English**
- **Greek**  

The language can be selected from the sidebar, and all text labels will adjust accordingly.

---

## Pagination

- The properties are displayed in paginated sections, with controls for navigating through pages.  
- You can adjust the number of properties per page and use the "Previous" and "Next" buttons for navigation.

---

## Notes

- Make sure that your CSV file includes the required columns for proper functionality.  
- Image URLs should be valid and point to publicly accessible images.  
- You can switch between languages using the language selection option in the sidebar.