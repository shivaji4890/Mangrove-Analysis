from flask import Flask,render_template, request,jsonify,redirect
import pandas as pd
import numpy as np
import datacube
from deafrica_tools.plotting import rgb, display_map
import datacube
import odc.algo
import matplotlib.pyplot as plt
from datacube.utils.cog import write_cog
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.plotting import display_map, rgb
import io
import base64
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("----------------------------------\n")
 
app = Flask(__name__)
# creating an instance of the Datacube class from the datacube module and assigns it to the variable dc.
dc = datacube.Datacube(app="04_Plotting")

image_base64=''
@app.route('/',methods=['POST','GET'])
def hello_world():
	
	return render_template("image.html")

@app.route('/about', methods=['POST','GET'])
def process():

	return render_template('about.html')
		
@app.route('/form',methods=['POST','GET'])
def form():
	image_base64=''
	
	total_area_array=[]
	pl_a=[]
	ml_a=[]
	ofm_array=[]
	dfm_array=[]
	sfm_array=[]
  
    #It checks if the HTTP request method is "POST". This condition ensures that the code block is only executed when the form is submitted using the HTTP POST method.
	if request.method=='POST' :
		# request.form.get() method is used to extract data from the submitted form.
		st= request.form.get('start-date')
		en = request.form.get('end-date')
		op = request.form.get('option')
		x = request.form.get('x')
		y = request.form.get('y')
		
		# json.loads() function deserializes the JSON strings stored in variables 'x' and 'y' into Python objects
		x_ti= json.loads(x)
		y_ti = json.loads(y)

		# Here we are iterating over the length of x_ti and performing the process inside it
		for i in range(len(x_ti)):
		
			lon_range = (x_ti[i][0],y_ti[i][0])
			lat_range = (x_ti[i][1],y_ti[i][1])

			time_range = (st,en)
			# dc.load() function is used to load specific data from a product called "s2a_sen2cor_granule" and with specified measurements and parameters.
			ds = dc.load(product="s2a_sen2cor_granule",
							measurements=["B04_10m","B03_10m","B02_10m", "B08_10m", "SCL_20m", "B11_20m", "B8A_20m", "B03_20m"],

						x=lon_range,
						y=lat_range,
						time=time_range,
						output_crs='EPSG:6933',
						resolution=(-30, 30))

			print(ds)
		

    # Get the spatial resolution of the dataset
			spatial_resolution = np.abs(ds.geobox.affine.a)

	# Calculate the area per pixel
			area_per_pixel = spatial_resolution**2

	# Determine the number of pixels in the dataset
			num_pixels = ds.sizes['x'] * ds.sizes['y']

	# Calculate the total area
			total_area = area_per_pixel * num_pixels


			total_area_km2=total_area/1000000
			total_area_array.append(total_area_km2)



			print("Area per pixel: {} square meters".format(area_per_pixel))
			print("Total area: {} square meters".format(total_area))
			print("Total area: {} square kms".format(total_area_km2))
			# convert dataset to float32 datatype so no-data values are set to NaN
			dataset =  odc.algo.to_f32(ds)
			# if there is no dataset present for the specified date and area
			if not ds:
				print("values:object")
				al_m='''
				The error can be :
				1. The range of coordinates are not applicable
				2. No data found in the range of dates
				
				Use high range date
				donot go beyond the range of map'''
				return render_template("image.html" ,al_m=al_m)
	
	        # if the selected analysis type is NDVI the the below code is executed
			if op=='NDVI':
				band_diff = dataset.B08_10m - dataset.B04_10m
				band_sum = dataset.B08_10m + dataset.B04_10m
				
				ds_index = band_diff/band_sum
				plt.figure()
				ds_index.plot()
				plt.xlabel('Value')
				plt.ylabel('Frequency')
				plt.title('Histogram')
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
			

				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()

				print('ndv')
			        
			
			

		# print(ndvi)
		# Generate the plot
		# for i in range(len(ds_index)):
			# plt.figure()
			band_diff = dataset.B08_10m - dataset.B04_10m
			band_sum = dataset.B08_10m + dataset.B04_10m
			ds_index=band_diff/band_sum
			dense_mangrove_mask = np.where((ds_index > 0.6) & (ds_index < 0.8), 1, 0)
			open_mangrove_mask = np.where((ds_index > 0.3) & (ds_index < 0.6) , 1, 0)
			x=np.where((ds_index>0.8) | (ds_index<0.1),1,0)

			sparse_mangrove_mask = np.where((ds_index > 0.1) & (ds_index < 0.3) , 1, 0)
			f_a=[]
			
			w=np.sum(x[0])
			print(np.sum(x[0]),"---",area_per_pixel)
			ta=area_per_pixel*w
			ta2=ta/1000000
			print(ta,ta2)
			time_values = ds_index.time.values
			d=['time','dfm','ofm','sfm','tfa']
			x=[]
			for i in range(len(dense_mangrove_mask)):
				w=[]
				w.append(pd.to_datetime(time_values[i]))
				w.append(area(dense_mangrove_mask[i],area_per_pixel))
				w.append(area(open_mangrove_mask[i],area_per_pixel))
				w.append(area(sparse_mangrove_mask[i],area_per_pixel))
				w.append(area(dense_mangrove_mask[i],area_per_pixel)+area(open_mangrove_mask[i],area_per_pixel)+area(sparse_mangrove_mask[i],area_per_pixel))
				f_a.append(area(dense_mangrove_mask[i],area_per_pixel)+area(open_mangrove_mask[i],area_per_pixel)+area(sparse_mangrove_mask[i],area_per_pixel))
				x.append(w)
	# df['time'] = pd.to_datetime(time_values)
			df = pd.DataFrame(x, columns=d)
			print(df)

# Assuming your time column is named 'time' and the value column is named 'ndvi'
# Convert the 'time' column to pandas timetime if it's not already in that format
# Read the CSV file into a pandas DataFrame

			df['time'] = pd.to_datetime(df['time'])
			print(df.head())
			X_train=df['time']
			y_train=df['ofm']
	# Split the data into training and test sets
	# X_train, X_test, y_train, y_test = train_test_split(df['time'], df['dfm'],test_size=0.3,shuffle=False)

	# Extract the time components as features
			X_train_features = pd.DataFrame()
			X_train_features['year'] = X_train.dt.year
			X_train_features['month'] = X_train.dt.month
			X_train_features['day'] = X_train.dt.day
			# Add more features as per your requirements

			# Initialize and fit the Random Forest Regressor model
			model = RandomForestRegressor()
			model.fit(X_train_features, y_train)

			# Extract features from the test data
			X_test_features = pd.DataFrame()
			X_test_features['year'] = [2018]
			X_test_features['month'] =[ 5]
			X_test_features['day'] = [5]
			# Add more features as per your requirements
			print(X_train_features.head())
			print(X_test_features.head())
			# Predict the values
			predictions = model.predict(X_test_features)

			# Print the predictions
			print(predictions)
			prede=model.predict(X_train_features)
			print(prede)
					
			if op=='NDWI':
				band_diff = dataset.B03_10m - dataset.B08_10m
				band_sum = dataset.B03_10m + dataset.B08_10m
				
				ds_index = band_diff/band_sum
				plt.figure(figsize=(10, 9))
				ds_index.plot()
				plt.xlabel('Value')
				plt.ylabel('Frequency')
				plt.title('Histogram')
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
			

				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')
				

				print('ndwi')
				buffer.close()
			
			if(len(ds['time'])==1):
				plt.figure()
				ds_index.plot( vmin=-1, vmax=1 ,figsize=(8, 4))

				# Convert the plot to a PNG image in memory
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
				image_base64_1 = base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()
			else:

				fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

				# Plot on the first subplot
				plot1 = ds_index[0].plot(ax=axes[0], vmin=-1, vmax=1)

				# Plot on the second subplot
				plot2 = ds_index[-1].plot(ax=axes[1], vmin=-1, vmax=1)
				
				axes[0].set_title(str(ds_index.time.values[0]).split('T')[0])
				axes[1].set_title(str(ds_index.time.values[-1]).split('T')[0])
				plt.subplots_adjust(wspace=0.3)

				# Convert the plot to a PNG image in memory
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
				image_base64_1 = base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()

			indices = np.arange(len(X_train))
			print("indices : : : ===  ",indices)


			if op=='ML PREDICTION':


			
			# Plot the actual values
				plt.figure()
				df['date']=pd.to_datetime(df['time'])
				#plt.plot(df['date'],f_a,marker='o')
				plt.plot(df['date'], df['ofm'], color='blue', label='Actual')
				plt.plot(df['date'], prede, color='red', label='Predicted')

				# Add labels and title
				plt.xlabel('Date')
				plt.ylabel('Mangrove_Area')
				plt.title('Random Forest Predictions - Actual vs. Predicted')

				# Add legend
				plt.legend()
				buffer=io.BytesIO()
				plt.savefig(buffer, format='png')

				buffer.seek(0)
				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()
				
		

				print('ndwi')
				buffer.close()
			
			
			ml_a.append(image_base64_2)
			pl_a.append(image_base64_1)
		
		
	print(len(pl_a),len(ml_a))
	
	return jsonify({'pl_a': pl_a,'ml_a':ml_a,'totala':total_area_km2})
def area( a ,area_per_pixel):
	print(np.sum(a))
	xw=area_per_pixel*np.sum(a)
	print(xw)
	return xw/1000000

def image_to_base64(image):
    plt.figure(figsize=(8,8))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64


if __name__ == '__main__':
	app.run(debug=True,port=5000,host='0.0.0.0')