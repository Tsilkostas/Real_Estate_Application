# Real Estate Property Catalog – Web Application

This is an interactive web application developed using **Python** and **Streamlit**, designed to serve as a property catalog for real estate listings. The application allows users to browse, filter, and view detailed property information using pre-prepared CSV data. It is targeted towards home buyers, renters, and agents who want to explore and analyze property listings efficiently.

## Features

- **Property Listing Grid**: View property listings in an attractive grid, showing key details such as the address, price, surface area, category, and a brief summary.
- **Pagination**: Navigate through large property datasets using a pagination system.
- **Filtering**: Filter listings based on:
  - Operation type (sale, rent, auction)
  - Price range
  - Surface area
  - Construction year
  - Features (e.g., parking, storage)
- **Search Functionality**: Search for properties by address or description keywords.
- **Interactive Map**: View property locations on an interactive map using **Folium**. Property markers display additional information when clicked.
- **Detailed Property View**: View detailed property information, including a large image, full description, and external URL.
- **Internationalization**: Support for English and Greek language toggle for UI elements.

## Data Input

The application reads real estate listings from a **CSV file** containing the following columns:

- `created_at`
- `updated_at`
- `category_en`
- `category_gr`
- `category_source_en`
- `type_gr`
- `operation`
- `lng`
- `lat`
- `surface`
- `construction_year`
- `price`
- `price_per_m2`
- `has_parking`
- `has_storage`
- `floor_num`
- `floor_cnt`
- `floor_min`
- `address_gr`
- `description_gr`
- `url`
- `img_url`
- `postcode`

Some fields like `lng`, `lat`, `surface`, `has_parking`, and `has_storage` may contain value errors, and data validation is performed during the file upload.

## Installation

### Prerequisites

Make sure to have **Python 3.11+** and the following Python packages installed:
- Streamlit
- Pandas
- Folium
- Other libraries are listed in the `requirements.txt` file.

### Steps to Set Up

1. **Clone the repository**:
   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/real-estate-app.git
