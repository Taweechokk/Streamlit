import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Functions for data cleansing methods
def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def filter_by_range(df, column, min_val, max_val):
    return df[(df[column] >= min_val) & (df[column] <= max_val)]

def delete_rows_by_datetime_range(df, column, start_datetime, end_datetime):
    # Convert the column to datetime if it's not already in datetime format
    df[column] = pd.to_datetime(df[column], errors='coerce')
    
    # Convert the input datetimes to pandas datetime objects
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)
    
    # Create a mask for rows where the datetime is within the range
    mask = (df[column] >= start_datetime) & (df[column] <= end_datetime)
    
    # Return the dataframe with rows outside the datetime range
    return df[~mask].reset_index(drop=True)

def display_logs():
    if st.session_state.logs:
        # Convert logs to DataFrame with the column name 'Logging'
        logs_df = pd.DataFrame(st.session_state.logs, columns=['Logging'])

        # Reverse the DataFrame to show the latest change on top
        logs_df = logs_df.iloc[::-1].reset_index(drop=True)

        # Add 1 to the index to start from 1
        logs_df.index += 1

        # Display the change log
        st.table(logs_df)

        # Button to delete the logging entries
        if st.button("Delete Logging"):
            st.session_state.logs = []  # Clear the logs
            st.success("Logging entries have been deleted.")
            # Simulate a "partial rerun" by only rerunning this fragment
            st.rerun()

    else:
        st.write("No changes have been logged yet.")

# Function to plot original and processed data
def plot_original_and_processed_data(original_dataframe, processed_dataframe, datetime_column, cleansing_column):
    """
    Plots original and processed data as subplots using Plotly and displays it in Streamlit.

    Parameters:
    original_dataframe (pd.DataFrame): The original dataframe containing data before cleansing.
    processed_dataframe (pd.DataFrame): The processed dataframe after cleansing.
    datetime_column (str): The name of the datetime column in the dataframes.
    cleansing_column (str): The column that is being visualized before and after cleansing.
    """
    
    # Create traces for the original and processed data
    trace1 = go.Scatter(
        x=original_dataframe[datetime_column],
        y=original_dataframe[cleansing_column],
        mode='lines+markers',
        name='Original Data',
        line=dict(color='blue'),
        marker=dict(size=6)
    )
    
    trace2 = go.Scatter(
        x=processed_dataframe[datetime_column],
        y=processed_dataframe[cleansing_column],
        mode='lines+markers',
        name='Processed Data',
        line=dict(color='green'),
        marker=dict(size=6)
    )

    # Create subplots with 2 rows
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Original Data', 'Processed Data'))

    # Add traces to the subplots
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=2, col=1)

    # Update the layout
    fig.update_layout(height=600, width=800, title_text="Subplots of Value1 and Value2 Over Time")

    # Update y-axis labels
    fig.update_yaxes(title_text="Original Dataframe Values", row=1, col=1)
    fig.update_yaxes(title_text="Processed Dataframe Values", row=2, col=1)

    # Update x-axis label
    fig.update_xaxes(title_text="Date and Time", row=2, col=1)

    # Display the figure in Streamlit using st.plotly_chart
    st.plotly_chart(fig, use_container_width=True)



# Function to save current state to history
def save_to_history(df):
    st.session_state.history.append(df.copy())
    st.session_state.redo_stack.clear()  # Clear redo stack after a new operation

# Undo function
def undo():
    # Ensure there is more than one history entry to prevent undoing beyond the original dataframe
    if len(st.session_state.history) > 1:
        st.session_state.redo_stack.append(st.session_state.processed_dataframe.copy())
        st.session_state.history.pop()
        st.session_state.processed_dataframe = st.session_state.history[-1]
        st.session_state.logs.append(f"Undo the last operation")
    else:
        st.session_state.processed_dataframe = st.session_state.history[0]

# Redo function
def redo():
    if st.session_state.redo_stack:
        st.session_state.history.append(st.session_state.processed_dataframe.copy())
        st.session_state.processed_dataframe = st.session_state.redo_stack[-1]
        st.session_state.redo_stack.pop()
        st.session_state.logs.append(f"Redo the last operation")

# Reset function to revert back to the original dataframe
def reset():
    st.session_state.processed_dataframe = st.session_state.original_dataframe.copy()
    st.session_state.history.clear()
    st.session_state.redo_stack.clear()
    save_to_history(st.session_state.original_dataframe)
    st.session_state.logs.append(f"Reset the dataframe to the original state")

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []  # Track past versions for undo
if 'redo_stack' not in st.session_state:
    st.session_state.redo_stack = []  # Track versions for redo
if 'processed_dataframe' not in st.session_state:
    st.session_state.processed_dataframe = None  # Store the processed dataframe
if 'logs' not in st.session_state:
    st.session_state.logs = [] # Store the logs

#start program
st.set_page_config(layout="wide")
st.sidebar.title("Navigation")
if "page" not in st.session_state:
    st.session_state.page = "File Upload"
page = st.sidebar.radio("Go to", ["File Upload", "Visualization","Correlation plot", "Export Processed File"], index=["File Upload", "Visualization", "Export Processed File"].index(st.session_state.page))
with st.sidebar:
    display_logs()
    
if page == "File Upload":
    st.title("File Upload Page")
    file_type = st.selectbox("Select File Type", ["AVEVA File","General File"])
    uploaded_file = st.file_uploader("Choose a file")
    
    if file_type == "General File" and uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        col1, col2 = st.columns(2)
        if col1.button("Submit"):
            dataframe.to_csv('uploaded_file.csv', index=False)
            st.success("General File has been submitted and saved successfully.")
            st.session_state['file_uploaded'] = True
            st.session_state.page = "Visualization"
        if col2.button("Cancel"):
            st.warning("File upload has been canceled.")
            st.session_state['file_uploaded'] = False
    
    elif file_type == "AVEVA File" and uploaded_file is not None:
        dataframe_original = pd.read_csv(uploaded_file)
        
        if 'df_head' not in st.session_state:
            st.session_state.head = None
        
        # Extract column names from the original dataframe
        head = dataframe_original.columns.tolist()
        # Create an empty dataframe with those column names and store it in session state
        st.session_state.df_head = pd.DataFrame(columns=head)
        
        
        dataframe_original.columns = dataframe_original.iloc[1].values
        dataframe = dataframe_original.iloc[4:, :]
        
    
        # Save the first 4 rows
        df_first_4_rows = dataframe_original.iloc[:4, :]
        st.session_state['df_first_4_rows'] = df_first_4_rows

        dataframe.rename(columns={'Extended Name':'DATETIME'}, inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
        dataframe = dataframe.apply(pd.to_numeric)
        dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])

        st.write("Preview of the uploaded AVEVA file:")
        st.write(dataframe)
        col1, col2 = st.columns(2)
        if col1.button("Submit"):
            dataframe.to_csv('uploaded_file.csv', index=False)
            st.success("AVEVA File has been submitted and saved successfully.")
            st.session_state['file_uploaded'] = True
        if col2.button("Cancel"):
            st.warning("File upload has been canceled.")
            st.session_state['file_uploaded'] = False
    
    if "file_uploaded" in st.session_state and st.session_state['file_uploaded']:
        st.success("File has been uploaded successfully.")
        st.session_state.page = "Visualization"
        original_dataframe = pd.read_csv('uploaded_file.csv')
        if 'original_dataframe' not in st.session_state:
            st.session_state.original_dataframe = original_dataframe.copy()
            st.session_state.processed_dataframe = original_dataframe.copy()
            if not st.session_state.history:
                save_to_history(st.session_state.original_dataframe)
                st.rerun()
            
elif page == "Visualization":
    st.title("Visualization Page")
    
    datetime_column = st.selectbox("Select the 'DATETIME' column for the x-axis:", st.session_state.original_dataframe.columns)
    cleansing_column = st.selectbox("Select the column to cleanse or filter:", st.session_state.original_dataframe.columns)
    
    plot_original_and_processed_data(st.session_state.original_dataframe, st.session_state.processed_dataframe, datetime_column, cleansing_column)

    # Undo/Redo/Reset Buttons aligned to the left in the same row
    col1, col2, col3, _ = st.columns([1, 1, 1, 15])  # Allocate more space for empty column to push buttons to the left
    with col1:
        if st.button("Undo", disabled=len(st.session_state.history) == 1):
            undo()
            st.rerun()  # Re-run to reflect the reset operation

    with col2:
        if st.button("Redo", disabled=len(st.session_state.redo_stack) == 0):
            redo()
            st.rerun()  # Re-run to reflect the reset operation

    with col3:
        if st.button("Reset", disabled=len(st.session_state.history) == 1):
            reset()
            st.rerun()  # Re-run to reflect the reset operation


    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Cleansing data time period", "Filter by Value Range",
                                             "Data Cleansing Outlier Methods","data logging","data","redo"])
                    
    with tab3:
        # Clean data filter by Outlier Methods
        st.subheader("Data Cleansing Outlier Methods")
        selected_methods = st.selectbox("Select the data cleansing methods to apply:", ["Z-score Method", "IQR Method"])

        if st.button("Confirm Process and Plot Graph"):
            if "Z-score Method" in selected_methods:
                st.session_state.processed_dataframe = remove_outliers_zscore(st.session_state.processed_dataframe, cleansing_column)
                st.session_state.logs.append(f"Applied Z-score method on {cleansing_column}")
                save_to_history(st.session_state.processed_dataframe)
                st.rerun()

            if "IQR Method" in selected_methods:
                st.session_state.processed_dataframe = remove_outliers_iqr(st.session_state.processed_dataframe, cleansing_column)
                st.session_state.logs.append(f"Applied IQR method on {cleansing_column}")
                save_to_history(st.session_state.processed_dataframe)
                st.rerun()
        else:
            st.info("The processed data will be shown after you confirm the process.")
            
    with tab2:
        # Filter by Value Range
        # if st.checkbox("Filter by Value Range", key="enabled_filter"):
            st.subheader("Filter by Value Range")
            
            # Get the min and max values for the cleansing column from the processed dataframe
            min_val = st.text_input(f"Enter minimum value for {cleansing_column}", value=str(st.session_state.processed_dataframe[cleansing_column].min()))
            max_val = st.text_input(f"Enter maximum value for {cleansing_column}", value=str(st.session_state.processed_dataframe[cleansing_column].max()))

            # Button to confirm and apply the filter
            if st.button("Confirm and Apply Filter"):
                if min_val and max_val:
                    try:
                        # Apply the filter
                        st.session_state.processed_dataframe = filter_by_range(st.session_state.processed_dataframe, cleansing_column, float(min_val), float(max_val))
                        st.session_state.logs.append(f"Filtered {cleansing_column} by range {min_val} - {max_val}")
                        st.success(f"Filtered {cleansing_column} by range {min_val} - {max_val}")
                        save_to_history(st.session_state.processed_dataframe)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error applying filter: {e}")
                else:
                    st.warning("Please provide both minimum and maximum values for filtering.")
    
    with tab1:
         # Clean data period time
            st.subheader("Set Date-Time Range")
            start_datetime = st.text_input("Enter start date-time \n_format: DD/MM/YY HH:MM:SS_  \n_example: 1/3/2023 10:00:00_", value="")
            end_datetime = st.text_input("Enter end date-time \n_format: DD/MM/YY HH:MM:SS_  \n_example: 1/3/2023 10:00:00_", value="")

            if 'saved_datetime_ranges' not in st.session_state:
                st.session_state['saved_datetime_ranges'] = []

            if st.button("Add Date-Time Range"):
                if start_datetime and end_datetime:
                    try:
                        # Assuming the function `replace_with_null_by_datetime_range` exists and works correctly
                        st.session_state['saved_datetime_ranges'].append(f"{start_datetime} to {end_datetime}")
                        st.session_state.logs.append(f"Added date-time range: {start_datetime} to {end_datetime}")
                        st.success(f"Added date-time range: {start_datetime} to {end_datetime}")
                    except Exception as e:
                        st.error(f"Error adding date-time range: {e}")
                else:
                    st.warning("Please provide both start and end date-times.")

            
            if st.session_state['saved_datetime_ranges']:
                st.subheader("Saved Date-Time Ranges for Replacement or Deletion")
                # Display the saved ranges and allow selection
                selected_ranges = []
                # Loop through the saved date-time ranges and create checkboxes
                for i, saved_range in enumerate(st.session_state['saved_datetime_ranges']):
                    cols = st.columns([4, 1])  # Two columns for the range and a delete button
                    with cols[0]:
                        # Add checkboxes for each saved range
                        if st.checkbox(saved_range, key=f"checkbox_{i}"):
                            selected_ranges.append(saved_range)
                    with cols[1]:
                        # Add a delete button for each saved range
                        if st.button(f"Delete", key=f"delete_{i}"):
                            st.session_state['saved_datetime_ranges'].pop(i)
                            st.success(f"Deleted range: {saved_range}")
                            st.rerun()  # To refresh the UI after deleting an item
                            
                if selected_ranges:
                    st.write(f"Selected ranges: {', '.join(selected_ranges)}")
                    
                    # Confirm button to clean the data using the selected ranges
                    if st.button("Confirm and Replace Data"):
                        try:
                            # Apply cleaning logic for each selected range
                            for range_str in selected_ranges:
                                start_datetime, end_datetime = range_str.split(" to ")
                                st.session_state.processed_dataframe = delete_rows_by_datetime_range(st.session_state.processed_dataframe, datetime_column, start_datetime, end_datetime)
                                st.session_state.logs.append(f"Replaced data between {start_datetime} and {end_datetime} with NULL")
                            
                            st.success(f"Replaced data in the selected date-time ranges: {', '.join(selected_ranges)}")
                            save_to_history(st.session_state.processed_dataframe)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error replacing data in the selected ranges: {e}")
                else:
                    st.write("Please select at least one saved range to confirm.")
                    
            else:
                st.write("No date-time ranges have been saved yet.")
    
    # with tab4:
        # display_logs()
        
    with tab5:
        st.write(st.session_state.history)
                
    with tab6:
        st.write(st.session_state.redo_stack)
        
# Scatter Plot page
elif page == "Correlation plot":
    st.title("Correlation plot")

    # Dropdown menus for selecting x and y columns for the scatter plot
    available_columns = st.session_state.processed_dataframe.columns.tolist()

    x_col = st.selectbox("Select X column", available_columns)
    y_col = st.selectbox("Select Y column", available_columns)

    # Create scatter plot based on session state DataFrame and selected columns
    fig = px.scatter(
        x=st.session_state.processed_dataframe[x_col],
        y=st.session_state.processed_dataframe[y_col],
        labels={
            'x': x_col,  # Set X-axis label dynamically
            'y': y_col   # Set Y-axis label dynamically
        },
        title=f'Scatter plot of {x_col} vs {y_col}'  # Optional: Add a title to the plot
        # size_max=10  # Maximum size of the marker
    )

    # Display plotly chart with selection event
    event = st.plotly_chart(fig, on_select="rerun")
    
    # Undo/Redo/Reset Buttons aligned to the left in the same row
    col1, col2, col3, _ = st.columns([1, 1, 1, 15])  # Allocate more space for empty column to push buttons to the left
    with col1:
        if st.button("Undo", disabled=len(st.session_state.history) == 1):
            undo()
            st.rerun()  # Re-run to reflect the reset operation

    with col2:
        if st.button("Redo", disabled=len(st.session_state.redo_stack) == 0):
            redo()
            st.rerun()  # Re-run to reflect the reset operation

    with col3:
        if st.button("Reset", disabled=len(st.session_state.history) == 1):
            reset()
            st.rerun()  # Re-run to reflect the reset operation

    # Check if any points are selected
    if event:
        if event.selection:
            #st.write("Selection data:", event.selection)

            # Extract the relevant 'x' and 'y' values based on the user's selected columns
            x = [point['x'] for point in event.selection["points"]]
            y = [point['y'] for point in event.selection["points"]]

            # Create a DataFrame with the extracted 'x' and 'y' values
            df2 = pd.DataFrame({
                x_col: x,  # dynamically set x column name
                y_col: y   # dynamically set y column name
            })
            # Set the index to start at 1
            df2.index = pd.Index(range(1, len(df2) + 1))
            st.write("Selected points:")
            st.write(df2)
            # Display the number of selected rows
            st.write(f"Number of selected points: {len(df2)}")
            
            # Filter out the selected points from the original DataFrame
            df1_filtered = st.session_state.processed_dataframe[
                ~(st.session_state.processed_dataframe[x_col].isin(x) & st.session_state.processed_dataframe[y_col].isin(y))
            ]

            # st.write("Filtered DataFrame:")
            # st.write(df1_filtered)

            # Create a button to confirm the changes and update the graph
            if st.button('Confirm and Update DataFrame'):
                # Update the DataFrame in session state
                st.session_state.processed_dataframe = df1_filtered
                save_to_history(st.session_state.processed_dataframe)
                st.session_state.logs.append(f"Deleted selected points {len(df2)}")
                # Rerun the script to update the plot with the new DataFrame
                st.rerun()
        else:
            st.write("No points selected.")
    else:
        st.write("No selection event triggered.")    
            
elif page == "Export Processed File":
    st.title("Export Processed File")
    if 'file_uploaded' in st.session_state and st.session_state['file_uploaded']:
        processed_dataframe = st.session_state.processed_dataframe

        # Merge the first 4 rows with the processed dataframe
        processed_dataframe2 = pd.concat([st.session_state['df_first_4_rows'], processed_dataframe], ignore_index=True)

        processed_dataframe2.loc[4:, 'Extended Name'] = processed_dataframe2.loc[4:, 'DATETIME']

        # # delete DATETIME column
        processed_dataframe2.drop(columns=['DATETIME'], inplace=True)
        

        processed_dataframe2.columns = st.session_state.df_head.columns
        processed_dataframe2 = pd.concat([st.session_state.df_head, processed_dataframe2], ignore_index=True)
        
        
        # Display the merged DataFrame to verify the first column values
        st.write("Preview of the final DataFrame with original first column:")
        st.write(processed_dataframe2)
        
        
         # Show the dimensions of the processed dataframe (rows, columns)
        num_rows, num_cols = processed_dataframe2.shape
        st.write(f"Number of rows: {num_rows}, Number of columns: {num_cols}")

        
        
        st.header("Download Processed File")
        buffer = BytesIO()
        processed_dataframe2.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Download Processed CSV",
            data=buffer,
            file_name="processed_file.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("Please upload and submit a file on the 'File Upload' page.")