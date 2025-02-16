import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium  
from folium.plugins import MarkerCluster
import ast
from typing import Dict, List, Any, Optional, Tuple

# Set up internationalization
def get_labels(language: str) -> Dict[str, str]:
    labels = {
        "English": {
            "title": "Real Estate Property Catalog",
            "upload_prompt": "Upload a CSV file",
            "success_upload": "File uploaded successfully!",
            "error_columns": "The uploaded file is missing required columns.",
            "info_upload": "Please upload a CSV file to proceed.",
            "filters": "Filters",
            "operation_type": "Operation Type",
            "price_range": "Price Range",
            "surface_area": "Surface Area",
            "showing_properties": "Showing {} properties",
            "page_number": "Page Number",
            "property_listings": "Property Listings",
            "no_image": "No valid image available",
            "image_error": "Image error: {}",
            "price": "Price",
            "category": "Category",
            "description": "Description",
            "property_map": "Property Map",
            "surface_area_unit": "Surface Area",
            "construction_year": "Construction Year",
            "original_listing": "Original Listing",
            "back_to_listings": "Back to Listings",
            "language": "Language",
            "view_details": "View Details",
            "no_price_data": "No valid price data available for filtering.",
            "no_surface_data": "No valid surface area data available for filtering.",
            "has_parking": "Has Parking",
            "has_storage": "Has Storage",
            "floor_number": "Floor Number",
            "sort_by": "Sort By",
            "rows_dropped": "{} rows were dropped due to missing or invalid data.",
            "created_at": "Created At",
            "updated_at": "Updated At",
            "postcode": "Postal Code",
            "yes": "Yes",
            "no": "No",
            "all": "All"
        },
        "Greek": {
            "title": "Κατάλογος Ακινήτων",
            "upload_prompt": "Ανεβάστε ένα αρχείο CSV",
            "success_upload": "Το αρχείο ανέβηκε με επιτυχία!",
            "error_columns": "Το αρχείο που ανεβάσατε δεν περιέχει τις απαιτούμενες στήλες.",
            "info_upload": "Παρακαλώ ανεβάστε ένα αρχείο CSV για να συνεχίσετε.",
            "filters": "Φίλτρα",
            "operation_type": "Τύπος Λειτουργίας",
            "price_range": "Εύρος Τιμής",
            "surface_area": "Εμβαδόν",
            "showing_properties": "Εμφάνιση {} ακινήτων",
            "page_number": "Αριθμός Σελίδας",
            "property_listings": "Λίστα Ακινήτων",
            "no_image": "Δεν υπάρχει διαθέσιμη εικόνα",
            "image_error": "Σφάλμα εικόνας: {}",
            "price": "Τιμή",
            "category": "Κατηγορία",
            "description": "Περιγραφή",
            "property_map": "Χάρτης Ακινήτων",
            "surface_area_unit": "Εμβαδόν",
            "construction_year": "Έτος Κατασκευής",
            "original_listing": "Αρχική Καταχώρηση",
            "back_to_listings": "Πίσω στη Λίστα",
            "language": "Γλώσσα",
            "view_details": "Προβολή Λεπτομερειών",
            "no_price_data": "Δεν υπάρχουν έγκυρα δεδομένα τιμών για φιλτράρισμα.",
            "no_surface_data": "Δεν υπάρχουν έγκυρα δεδομένα εμβαδού για φιλτράρισμα.",
            "has_parking": "Έχει Πάρκινγκ",
            "has_storage": "Έχει Αποθήκη",
            "floor_number": "Όροφος",
            "sort_by": "Ταξινόμηση κατά",
            "rows_dropped": "{} γραμμές αφαιρέθηκαν λόγω ελλιπών ή άκυρων δεδομένων.",
            "created_at": "Ημερομηνία Δημιουργίας",
            "updated_at": "Ημερομηνία Ενημέρωσης",
            "postcode": "Ταχυδρομικός Κώδικας",
            "yes": "Ναι",
            "no": "Όχι",
            "all": "Όλα"
        }
    }
    return labels.get(language, labels["English"])

# Cache the data loading
@st.cache_data
def load_data(uploaded_file):
    df_original = pd.read_csv(uploaded_file)
    
    # Store original row count
    original_count = len(df_original)
    
    # Convert numeric fields
    df_original["price"] = pd.to_numeric(df_original["price"], errors="coerce")
    df_original["surface"] = pd.to_numeric(df_original["surface"], errors="coerce")
    df_original["lat"] = pd.to_numeric(df_original["lat"], errors="coerce")
    df_original["lng"] = pd.to_numeric(df_original["lng"], errors="coerce")
    
    # Convert boolean fields
    for col in ["has_parking", "has_storage"]:
        df_original[col] = df_original[col].map(
            {True: True, False: False, "True": True, "False": False, 
             "true": True, "false": False, 1: True, 0: False}
        )
    
    # Drop rows with missing essential data
    df_clean = df_original.dropna(subset=["price", "surface", "lat", "lng"])
    
    # Calculate how many rows were dropped
    rows_dropped = original_count - len(df_clean)
    
    return df_clean, rows_dropped

def get_image_url(img_data: Any) -> Optional[str]:
    try:
        img_urls = ast.literal_eval(img_data) if isinstance(img_data, str) else []
        first_img = img_urls[0] if img_urls and isinstance(img_urls[0], str) and img_urls[0].startswith("http") else None
        return first_img
    except Exception:
        return None

def display_property_card(row: pd.Series, idx: int, labels: Dict[str, str]):
    with st.container():
        st.markdown("""
        <style>
        .property-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f9f9f9;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="property-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                first_img = get_image_url(row["img_url"])
                if first_img:
                    st.image(first_img, use_container_width=True)
                else:
                    st.info(labels["no_image"])

            with col2:
                st.subheader(row["address_gr"])
                st.write(f"**{labels['price']}:** {row['price']} €")
                st.write(f"**{labels['category']}:** {row['category_en']}")
                st.write(f"**{labels['surface_area_unit']}:** {row['surface']} m²")
                
                # Add parking and storage info
                parking = labels["yes"] if row["has_parking"] else labels["no"]
                storage = labels["yes"] if row["has_storage"] else labels["no"]
                st.write(f"**{labels['has_parking']}:** {parking} | **{labels['has_storage']}:** {storage}")
                
                # Add floor info if available
                if pd.notna(row["floor_num"]):
                    st.write(f"**{labels['floor_number']}:** {row['floor_num']}")
                
                description = str(row["description_gr"]) if pd.notna(row["description_gr"]) else ""
                st.write(f"**{labels['description']}:** {description[:100]}...")
                
                if st.button(labels["view_details"], key=f"view_{idx}"):
                    st.session_state["selected_property"] = row.to_dict()
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Sidebar for language selection first
    language = st.sidebar.selectbox("Language / Γλώσσα", ["English", "Greek"])
    labels = get_labels(language)
    
    # Title
    st.title(labels["title"])
    
    # File uploader
    uploaded_file = st.file_uploader(labels["upload_prompt"], type=["csv"])
    
    if uploaded_file is not None:
        # Load the CSV file with caching
        df, rows_dropped = load_data(uploaded_file)
        
        # Report dropped rows if any
        if rows_dropped > 0:
            st.warning(labels["rows_dropped"].format(rows_dropped))
        
        # Validate columns
        required_columns = [
            "created_at", "updated_at", "category_en", "category_gr", "category_source_en", 
            "type_gr", "operation", "lng", "lat", "surface", "construction_year", 
            "price", "price_per_m2", "has_parking", "has_storage", "floor_num", 
            "floor_cnt", "floor_min", "address_gr", "description_gr", "url", "img_url", "postcode"
        ]
        
        if all(column in df.columns for column in required_columns):
            st.success(labels["success_upload"])
            
            # Sidebar for filtering
            st.sidebar.header(labels["filters"])
            
            
            # Operation type filter
            operation_types = sorted(set(["Sale", "Auction"]) | set(df["operation"].unique()))
            # operation_type = st.sidebar.selectbox(labels["operation_type"], [labels["all"]] + operation_types)
            operation_type = st.sidebar.selectbox(
                labels["operation_type"],
                operation_types
            )
            # Category filter
            categories = [labels["all"]] + sorted(df["category_en"].unique().tolist())
            selected_category = st.sidebar.selectbox(labels["category"], categories)
            
            # Price range filter
            if not df["price"].empty:
                price_range = st.sidebar.slider(
                    labels["price_range"],
                    float(df["price"].min()),
                    float(df["price"].max()),
                    (float(df["price"].min()), float(df["price"].max()))
                )
            else:
                st.error(labels["no_price_data"])
                price_range = (0, 1)  # Default values to avoid errors
            
            # Surface area filter
            if not df["surface"].empty:
                surface_range = st.sidebar.slider(
                    labels["surface_area"],
                    float(df["surface"].min()),
                    float(df["surface"].max()),
                    (float(df["surface"].min()), float(df["surface"].max()))
                )
            else:
                st.error(labels["no_surface_data"])
                surface_range = (0, 1)  # Default values to avoid errors
            
            # Construction year filter if available
            if df["construction_year"].notna().any():
                min_year = int(df["construction_year"].min())
                max_year = int(df["construction_year"].max())
                year_range = st.sidebar.slider(
                    labels["construction_year"],
                    min_year,
                    max_year,
                    (min_year, max_year)
                )
            else:
                year_range = (0, 3000)  # Default wide range
            
            # Parking and storage filters
            parking_options = [labels["all"], labels["yes"], labels["no"]]
            has_parking = st.sidebar.radio(labels["has_parking"], parking_options)
            
            storage_options = [labels["all"], labels["yes"], labels["no"]]
            has_storage = st.sidebar.radio(labels["has_storage"], storage_options)
            
            # Sort options
            sort_options = ["price", "surface", "construction_year", "created_at"]
            sort_labels = {
                "price": labels["price"],
                "surface": labels["surface_area_unit"],
                "construction_year": labels["construction_year"],
                "created_at": labels["created_at"]
            }
            sort_by = st.sidebar.selectbox(
                labels["sort_by"],
                sort_options,
                format_func=lambda x: sort_labels.get(x, x)
            )
            sort_ascending = st.sidebar.checkbox("Ascending", value=True)
            
            # Apply filters
            filtered_df = df[df["operation"] == operation_type]
            
            # Apply category filter if not "All"
            if selected_category != labels["all"]:
                filtered_df = filtered_df[filtered_df["category_en"] == selected_category]
            
            # Apply other filters
            filtered_df = filtered_df[
                (filtered_df["price"] >= price_range[0]) & 
                (filtered_df["price"] <= price_range[1]) &
                (filtered_df["surface"] >= surface_range[0]) & 
                (filtered_df["surface"] <= surface_range[1]) &
                (
                    (filtered_df["construction_year"] >= year_range[0]) & 
                    (filtered_df["construction_year"] <= year_range[1]) |
                    filtered_df["construction_year"].isna()
                )
            ]
            
            # Apply parking filter
            if has_parking == labels["yes"]:
                filtered_df = filtered_df[filtered_df["has_parking"] == True]
            elif has_parking == labels["no"]:
                filtered_df = filtered_df[filtered_df["has_parking"] == False]
            
            # Apply storage filter
            if has_storage == labels["yes"]:
                filtered_df = filtered_df[filtered_df["has_storage"] == True]
            elif has_storage == labels["no"]:
                filtered_df = filtered_df[filtered_df["has_storage"] == False]
            
            # Apply sorting
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=sort_ascending)
            
            # Display filtered results count
            st.write(labels["showing_properties"].format(len(filtered_df)))
            
            # Detailed property view (if property is selected)
            if "selected_property" in st.session_state:
                property = st.session_state["selected_property"]
                
                st.header(property["address_gr"])
                
                first_img = get_image_url(property["img_url"])
                if first_img:
                    st.image(first_img, use_container_width=True)
                
                # Display all property details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{labels['price']}:** {property['price']} €")
                    st.write(f"**{labels['surface_area_unit']}:** {property['surface']} m²")
                    st.write(f"**{labels['construction_year']}:** {property['construction_year']}")
                    
                    # Boolean properties
                    parking = labels["yes"] if property["has_parking"] else labels["no"]
                    storage = labels["yes"] if property["has_storage"] else labels["no"]
                    st.write(f"**{labels['has_parking']}:** {parking}")
                    st.write(f"**{labels['has_storage']}:** {storage}")
                
                with col2:
                    # Floor information
                    if pd.notna(property["floor_num"]):
                        st.write(f"**{labels['floor_number']}:** {property['floor_num']}")
                    
                    # Category information
                    st.write(f"**{labels['category']}:** {property['category_en']} / {property['category_gr']}")
                    
                    # Dates
                    if pd.notna(property["created_at"]):
                        st.write(f"**{labels['created_at']}:** {property['created_at']}")
                    if pd.notna(property["updated_at"]):
                        st.write(f"**{labels['updated_at']}:** {property['updated_at']}")
                    
                    # Postcode
                    if pd.notna(property["postcode"]):
                        st.write(f"**{labels['postcode']}:** {property['postcode']}")
                
                # Full description
                st.write(f"**{labels['description']}:**")
                st.write(property['description_gr'])
                
                # Link to original listing
                st.write(f"**{labels['original_listing']}:** [Link]({property['url']})")
                
                if st.button(labels["back_to_listings"]):
                    del st.session_state["selected_property"]
                    st.rerun()
            
            else:
                # Pagination
                page_size = 10
                max_pages = max(1, len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0))
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    page_number = st.number_input(
                        labels["page_number"], 
                        min_value=1, 
                        max_value=max_pages,
                        value=1
                    )
                
                start_idx = (page_number - 1) * page_size
                end_idx = min(start_idx + page_size, len(filtered_df))
                
                # Display property listings in a grid
                st.header(labels["property_listings"])
                
                # Use the paginated dataframe
                paginated_df = filtered_df.iloc[start_idx:end_idx]
                
                for idx, row in paginated_df.iterrows():
                    display_property_card(row, idx, labels)
                
                # Pagination controls
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    if page_number > 1:
                        if st.button("← Previous"):
                            st.session_state["page_number"] = page_number - 1
                            st.rerun()
                
                with cols[2]:
                    if page_number < max_pages:
                        if st.button("Next →"):
                            st.session_state["page_number"] = page_number + 1
                            st.rerun()
                
                # Interactive map view
                st.header(labels["property_map"])
                filtered_df_map = filtered_df.dropna(subset=["lat", "lng"])
                if not filtered_df_map.empty:
                    center = [filtered_df["lat"].mean(), filtered_df["lng"].mean()]
                    m = folium.Map(location=center, zoom_start=12)
                    
                    # Add marker cluster
                    marker_cluster = MarkerCluster().add_to(m)
                
                    for idx, row in filtered_df_map.iterrows():
                        # Create popup content
                        popup_html = f"""
                        <strong>{row['address_gr']}</strong><br>
                        {labels['price']}: {row['price']} €<br>
                        {labels['surface_area_unit']}: {row['surface']} m²<br>
                        {labels['category']}: {row['category_en']}
                        """
                        
                        # Add marker to cluster
                        folium.Marker(
                            location=[row["lat"], row["lng"]],
                            popup=folium.Popup(popup_html, max_width=300),
                        ).add_to(marker_cluster)
                    
                    # Use st_folium to render the map
                    st_folium(m, width=700, height=500)
                else:
                     st.warning("No properties with valid coordinates found for the selected filters.")   
        
        else:
            st.error(labels["error_columns"])
    else:
        st.info(labels["info_upload"])

if __name__ == "__main__":
    main()