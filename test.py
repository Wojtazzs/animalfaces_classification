import data_func as func

svc_m = func.load_model("model.svc")

data = func.load_images_to_dict("images")
images = data['data']
x_img = func.extract_features_from_images(images)
y_img = data['label']

print(f"Accuracy svc: {svc_m.score(x_img, y_img)}")

print(func.test_single_image(filename="images/BearHead/BearHead_2.jpg", model=svc_m, visualization=True))

