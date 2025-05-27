import pickle
import json

# Adjust these if your actual filenames differ
with open('/kaggle/input/pickle-model/banglor_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('/kaggle/input/pickle-model/columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location.lower())  # make sure locations are lowercase in your json
    except ValueError:
        loc_index = -1  # Location not found

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)



import gradio as gr

# Create clean and user-friendly labels for locations
locations = [loc.title() for loc in data_columns[3:]]  # UI-friendly location list

# Dropdown options
sqft_options = [i for i in range(500, 5100, 100)]       # 500 to 5000 sqft
bath_options = list(range(1, 11))                       # 1 to 10 bathrooms
bhk_options = list(range(1, 11))                        # 1 to 10 BHK

def gradio_predict(location, sqft, bath, bhk):
    price = predict_price(location.lower(), sqft, bath, bhk)
    return f"Predicted Price: ₹{price} Lakhs"

iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(choices=locations, label="Location"),
        gr.Dropdown(choices=sqft_options, label="Total Square Feet"),
        gr.Dropdown(choices=bath_options, label="Number of Bathrooms"),
        gr.Dropdown(choices=bhk_options, label="Number of BHKs")
    ],
    outputs="text",
    title="🏠 Bangalore House Price Predictor",
    description="Select property details to get a predicted price (in Lakhs)."
)

iface.launch(share=True)
