import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium  
from folium.plugins import MarkerCluster
import ast
import re
from typing import Dict, Any, Optional

# Set up internationalization
def get_labels(language: str) -> Dict[str, str]:
    """ 
    Returns a dictionary of labels (UI text) for the selected language.

    Args:
        language (str): The selected language (e.g., "English" or "Greek").

    Returns:
       Dict[str, str]: A dictionary of labels for the selected language.
    """
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
            "all": "All",
            "search_properties": "Search Properties",
            "search_placeholder": "Enter address, keywords, or features...",
            "search_results": "Found {} properties matching '{}'",
            "no_search_results": "No properties found matching '{}'",
            "advanced_search": "Advanced Search",
            "search_in": "Search in:",
            "search_address": "Address",
            "search_description": "Description",
            "search_both": "Both",
            "recent_searches": "Recent Searches"
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
            "all": "Όλα",
            "search_properties": "Αναζήτηση Ακινήτων",
            "search_placeholder": "Εισάγετε διεύθυνση, λέξεις-κλειδιά, ή χαρακτηριστικά...",
            "search_results": "Βρέθηκαν {} ακίνητα που ταιριάζουν με '{}'",
            "no_search_results": "Δεν βρέθηκαν ακίνητα που ταιριάζουν με '{}'",
            "advanced_search": "Προχωρημένη Αναζήτηση",
            "search_in": "Αναζήτηση σε:",
            "search_address": "Διεύθυνση",
            "search_description": "Περιγραφή", 
            "search_both": "Και τα δύο",
            "recent_searches": "Πρόσφατες Αναζητήσεις"
        }
    }
    return labels.get(language, labels["English"])

# Helper function to highlight search terms in text
def highlight_text(text, query):
    if not query or not isinstance(text, str):
        return text
    
    highlighted = text
    for keyword in query.lower().split():
        pattern = re.compile(f'({re.escape(keyword)})', re.IGNORECASE)
        highlighted = pattern.sub(r'<span style="background-color: #ffff99;">\1</span>', highlighted)
    
    return highlighted

# Cache the data loading
@st.cache_data
def load_data(uploaded_file):
    """
    Loads and preprocesses the uploaded CSV file.

    Args:
        uploaded_file (UploadedFile): The CSV file uploaded by the user.

    Tuple[pd.DataFrame, int]: A tuple containing the cleaned DataFrame and the number of rows dropped.
    """
    df_original = pd.read_csv(uploaded_file)
    
    # Store original row count
    original_count = len(df_original)
    
    # Convert numeric fields (e.g., price, surface, lat, lng) to numeric types.
    # Invalid values (e.g., non-numeric strings) will be converted to NaN.
    df_original["price"] = pd.to_numeric(df_original["price"], errors="coerce")
    df_original["surface"] = pd.to_numeric(df_original["surface"], errors="coerce")
    df_original["lat"] = pd.to_numeric(df_original["lat"], errors="coerce")
    df_original["lng"] = pd.to_numeric(df_original["lng"], errors="coerce")
    
    # Convert boolean fields (e.g., has_parking, has_storage) to boolean types.
    # Handle various input formats (e.g., "True", "False", 1, 0).
    for col in ["has_parking", "has_storage"]:
        df_original[col] = df_original[col].map(
            {True: True, False: False, "True": True, "False": False, 
             "true": True, "false": False, 1: True, 0: False}
        )
    
    # Drop rows with missing essential data (e.g., price, surface, lat, lng).
    # These fields are required for filtering and map rendering.
    df_clean = df_original.dropna(subset=["price", "surface", "lat", "lng"])
    
    # Calculate how many rows were dropped due to missing or invalid data.
    rows_dropped = original_count - len(df_clean)
    
    return df_clean, rows_dropped

def get_image_url(img_data: Any) -> Optional[str]:
    try:
        img_urls = ast.literal_eval(img_data) if isinstance(img_data, str) else []
        first_img = img_urls[0] if img_urls and isinstance(img_urls[0], str) and img_urls[0].startswith("http") else None
        return first_img
    except Exception:
        return None

def display_property_card(row: pd.Series, idx: int, labels: Dict[str, str], search_query: str = ""):
    with st.container():
        st.markdown("""
        <style>
        .property-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f9f9f9;
            width: 100%;
            box-sizing: border-box;
        }
        @media (max-width: 600px) {
            .property-card {
                padding: 5px;
            }
        }
        .highlight {
            background-color: #ffff99;
            padding: 0 2px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="property-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                first_img = get_image_url(row["img_url"])
                if first_img:
                    st.image(first_img, use_container_width=True)  # Make image responsive
                else:
                    st.info(labels["no_image"])

            with col2:
                # Highlight address if it matches search query
                address_display = row["address_gr"]
                if search_query:
                    address_display = highlight_text(address_display, search_query)
                    st.markdown(f"### {address_display}", unsafe_allow_html=True)
                else:
                    st.subheader(address_display)
                
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
                
                # Highlight description if it matches search query
                description = str(row["description_gr"]) if pd.notna(row["description_gr"]) else ""
                description_preview = description
                
                if search_query:
                    description_preview = highlight_text(description_preview, search_query)
                    st.markdown(f"**{labels['description']}:** {description_preview}", unsafe_allow_html=True)
                else:
                    st.write(f"**{labels['description']}:** {description_preview}")
                
                # Store the selected property in session state to persist it across reruns.
                # This allows the app to display the detailed view when the user clicks "View Details"
                if st.button(labels["view_details"], key=f"view_{idx}"):
                    st.session_state["selected_property"] = row.to_dict()
                    st.session_state["search_query"] = search_query  # Save search query for highlighting in detail view
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Custom CSS for responsive property cards
    st.markdown("""
    <style>
    .property-card {
    border: 1px solid #ddd; /* Light gray border */
    border-radius: 5px; /* Rounded corners */
    padding: 10px; /* Spacing inside the card */
    margin-bottom: 10px; /* Spacing between cards */
    background-color: #f9f9f9; /* Light gray background */
    width: 100%; /* Full width */
    box-sizing: border-box; /* Include padding in width calculation */
    }
    .stImage img {
        max-width: 100%;
        height: auto;
    }
    @media (max-width: 600px) {
        .stImage img {
            width: 100%;
        }
        .property-card {
            padding: 5px;
        }
    }
    .highlight {
        background-color: #ffff99;
        padding: 0 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for search history
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

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
            
            # Search functionality in sidebar
            st.sidebar.header(labels["search_properties"])
            search_query = st.sidebar.text_input(
                label="Search", 
                placeholder=labels["search_placeholder"],
                label_visibility="hidden"
            )
            
            # Advanced search options
            with st.sidebar.expander(labels["advanced_search"], expanded=False):
                search_in = st.radio(
                    labels["search_in"],
                    [labels["search_both"], labels["search_address"], labels["search_description"]]
                )
            
            # Display recent searches
            if st.session_state.search_history:
                with st.sidebar.expander(labels["recent_searches"]):
                    for idx, hist_search in enumerate(st.session_state.search_history):
                        if st.button(hist_search, key=f"hist_{idx}"):
                            search_query = hist_search
                            st.rerun()
            
            # Add current search to history
            if search_query and search_query not in st.session_state.search_history:
                st.session_state.search_history.insert(0, search_query)
                # Keep only the 5 most recent searches
                st.session_state.search_history = st.session_state.search_history[:5]
                # Reset to page 1 when new search is performed
                if "page_number" in st.session_state:
                    st.session_state["page_number"] = 1
            
            # Sidebar for filtering
            st.sidebar.header(labels["filters"])
            
            # Operation type filter
            operation_types = sorted(set(["Sale", "Auction"]) | set(df["operation"].unique()))
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
            
            # Apply search filter based on user selection (address, description, or both)
            if search_query:
                # Determine where to search based on user selection
                if search_in == labels["search_address"]:
                    search_mask = filtered_df["address_gr"].str.lower().str.contains(search_query.lower(), na=False)
                elif search_in == labels["search_description"]:
                    search_mask = filtered_df["description_gr"].str.lower().str.contains(search_query.lower(), na=False)
                else:  # Default: search both
                    search_mask = (
                        filtered_df["address_gr"].str.lower().str.contains(search_query.lower(), na=False) |
                        filtered_df["description_gr"].str.lower().str.contains(search_query.lower(), na=False)
                    )
                
                # Apply the search filter
                filtered_df = filtered_df[search_mask]
                
                # Show search results message
                if len(filtered_df) > 0:
                    st.success(labels["search_results"].format(len(filtered_df), search_query))
                else:
                    st.warning(labels["no_search_results"].format(search_query))
            
            # Apply sorting
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=sort_ascending)
            
            # Display filtered results count (if not already shown by search)
            if not search_query:
                st.write(labels["showing_properties"].format(len(filtered_df)))
            
            # Detailed property view (if property is selected)
            if "selected_property" in st.session_state:
                property = st.session_state["selected_property"]
                search_query = st.session_state.get("search_query", "")
                
                # Display property address with highlighting if needed
                if search_query and "address_gr" in property:
                    highlighted_address = highlight_text(property["address_gr"], search_query)
                    st.markdown(f"# {highlighted_address}", unsafe_allow_html=True)
                else:
                    st.header(property["address_gr"])
                
                first_img = get_image_url(property["img_url"])
                if first_img:
                    st.image(first_img, width=300)
                
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
                
                # Full description with highlighting if needed
                st.write(f"**{labels['description']}:**")
                if search_query and "description_gr" in property and isinstance(property["description_gr"], str):
                    highlighted_description = highlight_text(property["description_gr"], search_query)
                    st.markdown(highlighted_description, unsafe_allow_html=True)
                else:
                    st.write(property['description_gr'])
                
                # Link to original listing
                st.write(f"**{labels['original_listing']}:** [Link]({property['url']})")
                
                if st.button(labels["back_to_listings"]):
                    del st.session_state["selected_property"]
                    if "search_query" in st.session_state:
                        del st.session_state["search_query"]
                    st.rerun()
            
            else:
                # Pagination setup
                page_size = 10
                max_pages = max(1, len(filtered_df) // page_size + (2 if len(filtered_df) % page_size > 0 else 0))

                # Initialize session state for page_number if it doesn't exist yet
                if "page_number" not in st.session_state:
                    st.session_state["page_number"] = 1

                # Pagination columns
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    page_number = st.number_input(
                        labels["page_number"], 
                        min_value=1, 
                        max_value=max_pages,
                        value=st.session_state["page_number"],
                        key="page_number_input"
                    )
                
                # Update session state for page_number if it changes through the number input
                if page_number != st.session_state["page_number"]:
                    st.session_state["page_number"] = page_number

                # Display current page info (Page X of Y)
                st.write(f"Page {st.session_state['page_number']} of {max_pages}")
                
                start_idx = (st.session_state["page_number"] - 1) * page_size
                end_idx = min(start_idx + page_size, len(filtered_df))
                
                # Display property listings in a grid
                st.header(labels["property_listings"])
                
                # Use the paginated dataframe
                paginated_df = filtered_df.iloc[start_idx:end_idx]
                
                for idx, row in paginated_df.iterrows():
                    display_property_card(row, idx, labels, search_query)
                
                # Pagination controls
                cols = st.columns([1, 1, 1])
                
                # Previous button
                with cols[0]:
                    if st.session_state["page_number"] > 1:
                        if st.button("← Previous"):
                            st.session_state["page_number"] -= 1
                            st.rerun()  # This triggers a rerun of the app
                    else:
                        st.button("← Previous", disabled=True)
                
                # Next button
                with cols[2]:
                    if st.session_state["page_number"] < max_pages:
                        if st.button("Next →"):
                            st.session_state["page_number"] += 1
                            st.rerun()  # This triggers a rerun of the app
                    else:
                        st.button("Next →", disabled=True)
                
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