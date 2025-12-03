# Applying CRISP-DM Methodology to the PuzzArm Project

**Document Purpose:**  
This document applies the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology to the PuzzArm project, structuring the AI/ML components (e.g., image identification for puzzle pieces, pose estimation, and imitation learning for arm control) across its six phases. CRISP-DM provides an iterative, non-linear framework for ML projects, emphasising business alignment and continuous refinement. This application serves as a checkpoint for the clustered AI course, mapping to ICTAII501 (designing AI solutions) and ICTAII502 (implementing ML models).

Use this as Assessment template, filling in project-specific details based on your work (e.g., Roboflow training/Arm training ).

**Project Recap:** PuzzArm is an AI-powered robotic system using Jetson Nano and xArm1S to solve a number puzzle (0-9 pieces), with dual-arm teleop (or similar method) for data collection. The ML focus is on vision-based detection, classification, and motion policies.

**Iteration Note:** CRISP-DM is cyclical—after Deployment, loop back to Business Understanding for refinements (e.g., adding new puzzles).



---

## Phase 1: Business Understanding
**Objective:** Define the problem, goals, and success criteria in business terms. Assess resources and risks.  

- **Business Problem:** Automate puzzle solving to create an educational robotics demo for expo demonstrations  and marketing events.
- **Data Mining Goals:** Develop models for piece detection (~70% accuracy), pose estimation (handling rotations), and arm control (50% pick-place success).  
- **Project Plan:** Timeline (4-6 weeks); resources (Jetson Nano, xArm1S, Roboflow). 
- **Risks:**
	- Compatibility issues with the chosen hardware challenging the viability of the original plan
  - time constraints not allowing for sufficient data collection, leading to a poor model accuracy
  - Minimum viable product is unable to be complete on time due to unforseen complications


- ***Student Input:*** [Describe how you  addressed the  business need 100-200 words]  

  My goal for addressing the business need was to create a classification model that can recognise the shape and positioning of the puzzle board and pieces and provide output to determine the precise movements necessary for the xArm to pick up pieces and put them in the correct slot. The key questions to answer in the data understanding phase are how accurate can the model be expected to be, and how can the dataset be improved to get better accuracy out of the model. The criteria for success will be a model with an F1 score of at least 0.7.
  
   *Mapping to Units:** ICTAII501 PC 1.1-1.2 (confirm work brief via CRISP-DM business phase).*

---

## Phase 2: Data Understanding
**Objective:** Collect initial data, explore it, and identify quality issues.  

- **Initial Data Collection:** 100-200 images of puzzle pieces/slots from top-down camera (via Jetson CSI), plus teleop videos (ROS2 bags) for joint states. Sources: Manual photos, Roboflow public datasets for augmentation.  
- **Data Description:** Structure (images: RGB, 224x224; labels: 0-9 classes; joints: 6D floats). Volume: ~5k samples post-augmentation.  
- **Data Exploration:** Use pandas/matplotlib for histograms (e.g., class balance: 10% per digit); identify issues (e.g., lighting bias via correlation plots).  
- **Student Input:** In order to have time to achieve a model that can give some sort of output for a minimum viable product, I only had time to collect data for one class, so my dataset only contained 1 class and 31 images
  ```python
  image_file_names = glob.glob('dataset_grip_2/*/*.jpg')

  image_file_names = pd.Series(image_file_names)

  images_df = pd.DataFrame()
  images_df['Filename'] = image_file_names.map(lambda img_name: img_name.split("\\")[2])
  images_df['ClassId'] = image_file_names.map(lambda img_name: int(img_name.split("\\")[1]))

  images_df
  ```  
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filename</th>
      <th>ClassId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[192, 134, 48, 637, 277, 621].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[192, 347, 129, 635, 272, 588].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[192, 495, 43, 634, 268, 516].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[192, 544, 217, 614, 240, 541].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[192, 571, 235, 635, 253, 499].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[192, 631, 161, 688, 307, 483].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[192, 703, 43, 629, 271, 614].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[192, 891, 154, 651, 289, 463].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[193, 284, 43, 662, 292, 347].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[193, 448, 222, 612, 238, 540].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[193, 487, 154, 657, 283, 586].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[193, 493, 64, 634, 274, 515].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[193, 494, 192, 634, 262, 519].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[193, 531, 181, 706, 306, 496].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[193, 532, 181, 706, 306, 496].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[193, 599, 161, 694, 308, 371].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[193, 602, 143, 634, 272, 545].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[193, 733, 103, 634, 275, 547].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[193, 806, 184, 672, 296, 347].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[194, 318, 106, 634, 274, 515].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[194, 342, 43, 637, 277, 402].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>21</th>
      <td>[194, 417, 151, 646, 275, 409].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>22</th>
      <td>[194, 425, 154, 644, 274, 482].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>23</th>
      <td>[194, 439, 70, 634, 274, 434].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>24</th>
      <td>[194, 446, 70, 634, 276, 352].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>25</th>
      <td>[194, 483, 103, 634, 273, 478].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>26</th>
      <td>[194, 499, 248, 678, 277, 482].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>27</th>
      <td>[194, 584, 181, 688, 301, 482].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>28</th>
      <td>[194, 682, 161, 693, 311, 360].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>29</th>
      <td>[194, 806, 155, 672, 305, 392].jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>30</th>
      <td>[195, 677, 241, 634, 258, 508].jpg</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>
 
*Mapping to Units ICTAII502 PC 1.1-1.6 (analyse requirements and data attributes using CRISP-DM data phase).* 
---

## Phase 3: Data Preparation
**Objective:** Clean, transform, and construct the final dataset for modeling.  

- **Data Cleaning:** Remove duplicates/blurry images (OpenCV thresholding); handle missing labels via Roboflow auto-annotation.  
- **Feature Engineering:** Augment for rotations (0-360° via Albumentations); normalize images (0-1 scale); engineer joint deltas from teleop recordings.  
- **Final Dataset:** Train (70%): 3.5k samples; Val (20%): 1k; Test (10%): 500. Format: PyTorch DataLoader for Jetson training.  
- **Student Input:** Applied colour jitter (random brightness, saturation, hue adjustments) to fix consistent ambience in shots and account for different environments, resized to 224x224 to allow input to resnet model, converted to tensor for format consistency, normalised with mean of [0.485, 0.456, 0.406], and standard deviation of [0.229, 0.224, 0.225]. Final number of samples: 31.
- **Mapping to Units:** ICTAII502 PC 2.1-2.4 (set parameters, engineer features per CRISP-DM prep phase).  

---

## Phase 4: Modeling
**Objective:** Select and apply ML techniques, tuning parameters.  

- **Model Selection:** - *Student input* - Resnet18, CNN, pretrained
- **Techniques Applied:** - *Student input* - supervised learning, classification
- **Model Building:**  - *Student input* - retrain final layer on my dataset, train detection (input: images, output: class). Export to TensorRT.  

*Mapping to Units ICTAII502 PC 3.1-3.5 (arrange validation, refine parameters via CRISP-DM modeling).*  

---

## Phase 5: Evaluation
**Objective:** Assess model performance against business goals; review process.  

- **Model Assessment:** - *Student input* - 
testing process during training:
``` python
     test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print('%d: %f' % (epoch, test_accuracy))
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy
```
output: 
0: 1.000000
1: 1.000000
2: 1.000000
3: 1.000000
4: 1.000000
5: 1.000000
6: 1.000000
7: 1.000000
8: 1.000000
9: 1.000000
10: 1.000000
11: 1.000000
12: 1.000000
13: 1.000000
14: 1.000000
15: 1.000000
16: 1.000000
17: 1.000000
18: 1.000000
19: 1.000000
20: 1.000000
21: 1.000000
22: 1.000000
23: 1.000000
24: 1.000000
25: 1.000000
26: 1.000000
27: 1.000000
28: 1.000000
29: 1.000000

- **Business Criteria Check:** Does it enable full puzzle solve <5 min?  - No. The model can only detect whether it is in a correct position for grabbing the number 8 piece, and the only output is the probability that it is in the correct position. 
- **Process Review:** Data quality issues? (e.g., rotations fixed via augments). Next iteration: - The small amount of data trained on and the overly consistent conditions the photos were taken in have led to overfitting, as demonstrated by the accuracy score being at 100% the entire training process. Next iteration, more data should be collected with different backdrops, and extra preprocessing steps should be applied.

*Mapping to Units ICTAII502 PC 5.1-5.6 (finalize evaluations, document metrics per CRISP-DM eval phase); ICTAII501 PC 3 (document design outcomes).*  

---

## Phase 6: Deployment
**Objective:** Plan rollout, monitoring, and maintenance.  

- **Deployment Plan:** *Student input* - For now, deployed to PC (cpu only) using a webcam for input. Can be deployed to Jetson in future if solution for issues with xArm module access is found.
- **Monitoring:** *Student input* - retrain quarterly and monitor grip success rate over recorded attempts.
- **Business Reporting:** *Student input*  - Demo video; report ROI (e.g. What it can do for the time invested). Maintenance: Version models in GitHub/Gitlab
- ROI report: The overall time spent was 5 weeks, including attempts/failure to get jetbot working for data collection. The result is a model that can return the probability that the arm is positioned to grab a piece, although the highest confidence achieved during model testing was around 0.6, which in most cases is not enough to actually be positioned correctly. 
- The purpose of the demonstration code, which is what is run in the demo video, is to show the puzzarm robot attempting to grab the puzzle piece when it detects a probability of at least 0.6 (as mentioned before, highest confidence achieved throughout testing). Purely for demonstration purposes, the servo values are adjusted incrementally by 1 until it achieves this (if not immediately after state is set to 'ready to grab'), which will in future need to be replaced with some kind of intuitive adjustment.
*Mapping to Units ICTAII501 PC 2 (design for deployment); ICTAII502 PC 4.1-4.5 (finalize test procedures).*  

---

## Overall Reflection and Iteration Plan
 **Next Steps:** *Student input* - What do you need to do next to achieve the project.  200 -400 words + code samples if required.

The significant overfitting problem the current result has is likely due to training a complex model on a very small dataset, so the first thing that will be required to achieve the project will be solving the overfitting problem by using a larger dataset, which will involve recording more image/joint pairs (aim for 100 for each piece for full dataset?). Aiming for more variety in the data collected, i.e. taking pictures against different backdrops, lighting etc., will also be beneficial. The resulting amount of data should be sufficient for more effectively fine-tuning the pretrained resnet18 model.
While this alone will significantly improve the model's ability to determine the right grip angle, it doesn't solve the problem that the current output is insufficient for determining the adjustments needed to reposition. Getting this working will require significant changes to the modeling process, as the current training code was originally designed for object avoidance, for which there are only two simple instructions to choose from depending on the output probability.
The whole process will then need to be repeated for every step in the puzzle process, including slot navigation and placement, in order to get the robot to be able to solve the full puzzle.


