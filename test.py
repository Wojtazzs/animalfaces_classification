import data_func as func
import time

# Loading the model
svc_m = func.load_model("model.svc")

# Example test/usage
# prints the prediction, visualization=True plots the image with the prediction on matplotlib
#print(func.test_single_image(filename="test_img.jpg", model=svc_m, visualization=False))

# Example of how to plot the changes of the image to see the feature extraction
func.plot_image_changes("test_img.jpg", model=svc_m)
