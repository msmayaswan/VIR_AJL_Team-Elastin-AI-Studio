Equitable AI for Dermatology

---

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Maya Swan | [@msmayaswan](https://github.com/msmayaswan) | Performed EDA, built inital model |
| Julia Husainzada | [@juliahusainzada](https://github.com/juliahusainzada) | Performed EDA, performed data augmentation |
| Isabella Alfaro | [@IsabellaAlfaro](https://github.com/IsabellaAlfaro) | Finetuned Xception model, improved accuracy score |
| Be-Once' Marsh | [@beonce](https://github.com/beonce) | Handled missing data |
| Michelle Cao | [@mcao694](https://github.com/mcao694) | Finetuning model | 

---

## **🎯 Project Highlights**

Built an Xception model using transfer learning and fine-tuning to accomplish the tas of building an ML model to classify skin conditions across diverse skin tones.
* Ranked 21st on the final Kaggle Leaderboard with a F1 score of 0.55188
* Used Tensorflow to interpret model decisions and improve performance
* Implemented data preprocessing to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* Clone the repo: git clone https://github.com/YOUR_GITHUB_USERNAME/equitable-ai-dermatology.git
cd equitable-ai-dermatology
* Install dependencies: pip install -r requirements.txt
* Set Up Kaggle API & Download Dataset: Sign in to Kaggle

Go to "Account Settings" > "API" and click Create New API Token

Download the kaggle.json file

Move it to ~/.kaggle/ (Linux/macOS) or C:\Users\<YourUsername>\.kaggle\ (Windows)

Run the following command to download the dataset: kaggle competitions download -c bttai-ajl-2025
unzip bttai-ajl-2025.zip -d data/

* Launch Jupyter Notebook to explore and run the model: jupyter notebook

---

## **🏗️ Project Overview**

Describe:
Our project is part of a Kaggle competition connected to the Break Through Tech AI Program, which focuses on advancing diversity and inclusion in artificial intelligence. This competition challenges participants to develop equitable AI solutions that address real-world biases in machine learning.

The Objective of the Challenge:
The goal of this challenge is to build a machine learning model that can accurately classify skin conditions across diverse skin tones. By ensuring that the model performs well across different skin types, we aim to create a more inclusive AI system that does not disproportionately favor lighter skin tones, a common issue in existing dermatology AI models.

Real-World Significance and Impact:
Skin condition detection plays a critical role in healthcare, influencing early diagnosis and treatment outcomes. However, AI models in dermatology often lack diverse training data, leading to biased predictions that fail to serve underrepresented populations. By participating in this challenge, we contribute to advancing equity in AI by developing a model that centers on those historically excluded in medical AI applications. Our work has the potential to improve access to accurate dermatological assessments for all individuals, regardless of their skin tone.

---

## **📊 Data Exploration**

**Describe:**

* The dataset is a subset of the FitzPatrick17k dataset, a labeled collection of about 17,000 images depicting a variety of serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions with a range of skin tones scored on the FitzPatrick skin tone scale (FST). About 4500 images are in this set, representing 21 skin conditions out of the 100+ in the full FitzPatrick set.

* The data exploration and preprocessing involved loading the dataset from CSV files, creating full file paths for images by combining the label and md5hash columns, and encoding the categorical label column into numerical values using LabelEncoder(). Additionally, the dataset was split into 80% for training and 20% for validation. These steps helped prepare the data for use in training a machine learning model while ensuring compatibility and efficiency in the process.

### ⚖️ Label Distribution in Training Data: Skin Condition Sample Count
*Displays the number of training images per skin condition (x-axis: image count, y-axis: conditions sorted by frequency).*

<img src="EDA_Visualizations/labels1.png" alt="EDA Visualization" width="600">

**Insights**: 
* **Class Imbalance:** Dominant classes (e.g., squamous-cell-carcinoma) vastly outnumber rarer ones (e.g., basal-cell-carcinoma-morpheaform).
* **Model Impact:** An imbalanced dataset may bias the model toward well-represented classes, reducing accuracy for underrepresented labels.
* **Healthcare Implications:** Underrepresentation could lead to misdiagnoses in real-world scenarios if not properly addressed.

### ⚖️ Three Partition Distribution
*Displays how many images exist per Fitzpatrick skin tone (x-axis) and diagnosis label (color/hue).*

<img src="EDA_Visualizations/partition.png" alt="EDA Visualization" width="600">

**Insights**: 
* There are far more images for lighter skin tones (e.g., scale 2 or 3) than darker tones (e.g., scale 5 or 6). This imbalance can cause the model to perform better on lighter tones and poorly on darker ones, potentially perpetuating healthcare disparities.
* Addressing underrepresentation is crucial for fairness and reducing misdiagnosis in darker skin tones; this can be achieved through targeted data augmentation, balanced sampling, and model adjustments like class weighting and fine-tuning.

### ⚖️ Fitzpatrick Scale Distribution
*Compares the counts of images grouped into three overarching diagnostic categories—malignant, non-neoplastic, and benign.* 

<img src="EDA_Visualizations/fitzpatrick.png" alt="EDA Visualization" width="600">

**Insights:**
* **Imbalance in Samples:** Malignant conditions dominate, followed by non-neoplastic, with benign conditions being the least represented.
* **Model Bias:** A model may learn to better detect malignant features, reducing accuracy for benign and non-neoplastic conditions.
* **Clinical Implications:** Misclassification can lead to unnecessary treatments or missed diagnoses, impacting patient care.
* **Mitigation Strategies:** Use data augmentation, class weighting, or collect more benign samples to balance the dataset and improve model performance.

---

## **🧠 Model Development**

### **Base Model:**  
* **Xception Network** pre-trained on ImageNet (fine-tuned end-to-end).  

### **Custom Classification Head:**  
* **Global Average Pooling** layer.  
* **Fully Connected Layer**: 1024 units, ReLU activation.  
* **L2 Weight Regularization** (λ=0.01) to mitigate overfitting.  
* **50% Dropout** for improved generalization.  
* **Softmax Output Layer** (114 classes).  

### ⚙️ Training Configuration  
### **Optimization Setup:**  
* **Optimizer**: Adam (initial LR = 1e-4).  
* **Loss Function**: Sparse Categorical Crossentropy.  
* **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.2, patience=3).  
* **Early Stopping**: Monitored validation loss (patience=5, restores best weights).  

### **Regularization Strategy:**  
* **L2 Weight Decay** (λ=0.01).  
* **Dropout (50%)** in the classification head.  

### 📊 Training Performance  
### **Dataset Configuration:**  
* **Training/Validation Split**: 80% / 20% (stratified sampling).  
* **Batch Size**: 32.  
* **Training Epochs**: 20 (early stopping applied).  

### **Model Performance:**  
* **Training Accuracy**: 100%
* **Validation Accuracy**: ~58%

---

## **📈 Results & Key Findings**

### Performance Metrics
* **Training Accuracy:** The model achieved a **training accuracy of 100%**, indicating that it learned the training data effectively.​ The high training accuracy indicates that the model learned the training data effectively. However, the significant drop in validation accuracy suggests potential overfitting, where the model may not generalize well to unseen data.​
* **Validation Accuracy:** The **validation accuracy was approximately 58%**, suggesting a significant drop from the training accuracy. This disparity points to potential overfitting, where the model performs well on training data but struggles to generalize to unseen data.​
* **Kaggle Leaderboard Score:** The model secured the **21st position on the Kaggle leaderboard with a final F1 score of 0.55188**, placing us in roughly the 70th percentile among all teams.

### Performance Across Different Skin Tones:

An essential aspect of this competition was ensuring equitable performance across diverse skin tones. The dataset's imbalance—favoring lighter skin tones—likely impacted the model's effectiveness on underrepresented groups. Addressing this imbalance is crucial to prevent perpetuating existing biases in dermatological assessments.​

### Insights from Model Fairness Evaluation:

The dataset exhibited a significant class imbalance, with certain skin conditions and skin tones being underrepresented. This imbalance can lead to biased predictions, where the model performs better on well-represented classes and skin tones, potentially resulting in misdiagnoses for underrepresented groups.​
  
## **🖼️ Impact Narrative**

As Dr. Randi mentioned in her challenge overview, ***“Through poetry, art, and storytelling, you can reach others who might not know enough to understand what’s happening with the machine learning model or data visualizations, but might still be heavily impacted by this kind of work.”***

Inspired by Dr. Joy Buolamwini’s “Unmasking AI” and the principles of the Algorithmic Justice League, our team has taken deliberate steps to ensure our dermatology AI model works equitably for all skin tones. Here’s how:

### 🛡️ Ensuring Fair Model Evaluation: 
We established stratified validation sets based on Fitzpatrick skin tones to monitor model performance across different groups. This approach allows us to detect discrepancies and adjust our training strategy if our model underperforms on darker skin tones, thus guarding against potential misdiagnoses in real-world clinical settings.

### 🗣️ EDA for Accessible Communication
We turn our detailed data analysis and fairness measures into visualizations that tell a story. Specifically, these important patterns and imbalances in our data explains AI bias in a way that’s easy for both experts and the community to understand, allowing for meaningful conversation on these issues.

### 🌍 Broader Impact on Healthcare Equity: Dr. Joy Buolamwini’s “Unmasking AI”
Our work could change how AI is used in dermatology. By reducing bias in our data and models, we help cut healthcare disparities. In short, our goal is to ensure every patient- no matter their skin tone- gets an accurate and fair diagnosis.

This project is not just about building a better model- it’s about challenging the status quo, inspiring change, and fighting for algorithmic justice in healthcare.

---

## **🚀 Next Steps & Future Improvements**
### Model Limitations:
* Our model reaches 100% training accuracy but only ~58% on validation data, indicating it memorizes training examples without generalizing well.
* The Xception network pre-trained on ImageNet is optimized for everyday images—not close-up dermatological details. It may miss the subtle color gradients, textures, and lesion borders crucial for accurate skin condition classification.
 
### With More Time/ Resources:
* Collect more images, especially for underrepresented skin conditions among the 114 classes, to better reflect real-world diversity.
* Explore models pre-trained on dermatology-specific datasets or further fine-tune the current model on a clinically rich dataset to capture domain nuances.
* Experiment with sophisticated augmentation methods (e.g., mixup, CutMix) and consider ensembling multiple architectures for improved performance.
* Implement fairness evaluation frameworks to assess and ensure equitable performance across different demographic groups.
  
### Mitigation Strategies for Fairness:
* Data Augmentation: Increase the representation of underrepresented skin tones through techniques such as image rotation, scaling, and color adjustments.​
* Class Weighting: Adjust the loss function to assign higher weights to underrepresented classes, encouraging the model to pay more attention to these classes during training.​
* Transfer Learning with Diverse Datasets: Incorporate additional datasets that are more representative of diverse populations to improve the model's generalizability.

---

## **📄 References & Additional Resources**

* Buolamwini, J. (n.d.). Unmasking AI: AJL Official Action Guide. Algorithmic Justice League. Retrieved from
https://drive.google.com/file/d/1kYKaVNR_l7Abx2kebs3AdDi6TlPviC3q/view
* What does “fairness” mean for machine learning systems? (n.d.). Haas School of Business, University of California, Berkeley. Retrieved from
https://haas.berkeley.edu/wp-content/uploads/What-is-fairness_-EGAL2.pdf


