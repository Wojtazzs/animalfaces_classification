import data_func as func
from data_func import np, train_test_split, plt, hog, confusion_matrix, svm


# 1. Load the images from the folder and resize them to 100x100 grayscale =====
print("Step 1: Loading images..")
data = func.load_images_to_dict("images", 100)
images = np.array(data['data'])
#==============================================================================


# 2. Get the unique labels and organize the data ==============================
print("Step 2: Extracting labels..")
labels = np.unique(data['label'])
y_data = np.array(data['label'])
x_img = np.zeros(len(images))
#==============================================================================


# 3. Feature extraction =======================================================
print("Step 3: Extracting features..")
for iter, image in enumerate(images):
    val, image = hog(
        image, 
        cells_per_block=(3, 3), 
        pixels_per_cell=(8, 8), 
        visualize=True
        )
    
    if iter == 0:
        x_img = np.zeros((len(images), val.shape[0]))
    if iter == 502:
        plt.imshow(image)
        plt.title(f"{y_data[iter]}")

    x_img[iter] = val
#==============================================================================


# 4. Split the data into train and test sets ==================================
print("Step 4: Splitting data..")
x_train, x_test, y_train, y_test = train_test_split(
    x_img, 
    y_data, 
    test_size=0.2, 
    stratify=y_data,
    )
#==============================================================================


# 5. Train the model ==========================================================
print("Step 5: Creating model..")
# Done test accuracy on different models, 
# best settings: SVC kernel="poly", degree=3, C=5, tol=0.000_001
svc_m = svm.SVC(kernel="poly", degree=3, C=5, tol=0.000_001)
#==============================================================================


# 6. Trainibg the model =======================================================
print("Step 6: Training model..")
svc_m.fit(x_train, y_train)
print(f"Accuracy svc: {svc_m.score(x_test, y_test)}")
#==============================================================================


# 7: plot the results =========================================================
print("Step 7: Plotting results..")
y_pred = svc_m.predict(x_test)
cm = confusion_matrix(y_test, y_pred, labels=labels)
print(cm)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest')

ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        title="Confusion matrix",
        ylabel='True label',
        xlabel='Predicted label')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

#sum of true positives
suma_true = 0
suma_false = 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i == j:
            suma_true += cm[i, j]
        else:
            suma_false += cm[i, j]
        
print("GUESSED: ")
print(f"True: {suma_true}, False: {suma_false}")

plt.show()
#==============================================================================


# 8. Saving the model to pickle ===============================================
func.save_model(svc_m, "model.svc")
#==============================================================================