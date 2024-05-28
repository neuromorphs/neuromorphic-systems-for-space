# Low-Power Visual Attention in In-Space Assembly Using Relative Motion 📡🚀

Organisers:

**Giulia D'Angelo**, giulia.dangelo.1990@gmail.com

Department of Cybernetics, Czech Technical University in Prague

**Alexander Hadjiivanov**, alexander.hadjiivanov@esa.int

Advanced Concepts Team, European Space Agency


## Introduction 🌟

Welcome to the Telluride 2024 Workshop in beautiful Colorado! This tutorial will guide you through our exciting project on low-power visual attention for in-space assembly using relative motion. In-space assembly of spacecraft is a highly precise task requiring accurate pose estimation of neighboring modules. This involves determining their relative position and attitude during proximity operations, which is crucial for successful assembly.

## Project Overview 🚀🔧

### Objective 🎯

The primary goal of this project is to explore the use of event-based cameras and visual attention mechanisms to improve the precision and efficiency of in-space assembly. We aim to reduce redundancy and focus on relevant parts of the visual field, enhancing event-based relative motion estimation algorithms to initiate and successfully carry out in-space assembly operations.

### Why It Matters 🌌

In-space assembly is a cornerstone for the future of space exploration and satellite deployment. Efficiently assembling spacecraft components in orbit can significantly reduce costs and expand the capabilities of space missions. However, this process demands highly accurate visual feedback and control systems, especially during the final stages of docking and alignment.

## Key Concepts 📚

### Event-Based Cameras 🎥

Event-based cameras are cutting-edge sensors that detect changes in the visual field, or "events," rather than capturing frames at fixed intervals. This allows for highly sensitive motion detection, crucial for tasks requiring precise relative motion estimation. However, as the relative motion of the target object closely matches the motion of the camera, fewer events are generated, posing a challenge during the critical final stages of assembly.

### Visual Attention Mechanisms 👀

Visual attention mechanisms are inspired by biological systems, focusing computational resources on the most relevant parts of the visual field. By integrating these mechanisms with event-based cameras, we can enhance the efficiency and accuracy of pose estimation, particularly in scenarios where event generation decreases.

If you want to know more about attention mechanisms and where the model takes inspiration from, please refer to Giulia D'Angelo's talk for ONM:

[![Low-Power Visual Attention in In-Space Assembly](https://img.youtube.com/vi/vwT_3bNNStg/0.jpg)](https://www.youtube.com/watch?v=vwT_3bNNStg&ab_channel=OpenNeuromorphic)


### References
- [D’Angelo, Giulia, et al. "Event driven bio-inspired attentive system for the iCub humanoid robot on SpiNNaker." *Neuromorphic Computing and Engineering* 2.2 (2022): 024008](https://iopscience.iop.org/article/10.1088/2634-4386/ac7ab6).
- [Iacono, M., D’Angelo, G., Glover, A., Tikhanoff, V., Niebur, E., & Bartolozzi, C. (2019, November). Proto-object based saliency for event-driven cameras. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 805-812). IEEE.](https://ieeexplore.ieee.org/abstract/document/8967943?casa_token=fR5p7q6rp0YAAAAA:lIfq0Q1Qd2nw0rCW4AIpYZA9Lnj9wcNqjSyhGxmM1N3fOgl7D6EoQKt6m7i2FzwtDxY9IZEGLw)
- [Ghosh, S., D’Angelo, G., Glover, A., Iacono, M., Niebur, E., & Bartolozzi, C. (2022). Event-driven proto-object based saliency in 3D space to attract a robot’s attention. Scientific reports, 12(1), 7645.](https://www.nature.com/articles/s41598-022-11723-6)
- [D'Angelo, G. (2022). A bio-inspired attentive model to predict perceptual saliency in natural scenes. The University of Manchester (United Kingdom).](https://www.proquest.com/openview/5cef6fddfe963d20ec1d663d6de5ea4f/1?pq-origsite=gscholar&cbl=51922&diss=y)


### Relative Motion Estimation Algorithms 🔄

Relative motion estimation involves calculating the disparity in motion between neighboring target modules or structures. This enables precise alignment and docking of spacecraft components through visual feedback. By combining event-based cameras with advanced motion estimation algorithms, we aim to improve the robustness of these systems in dynamic space environments.


## 📋 What You Need to Do

In this project, you will use the visual attention mechanism to detect the presence of teh aircraft.
We want you to create an innovative algorithm for in-space assembly and/or correct landing on a surface.
You can use any mechanism or algorithm you think is relevant to solve the problem.
You are strongly encouraged to explore other visual attention mechanisms if you prefer.
Your task is to come up with a solution that ensures precise pose estimation and alignment of spacecraft components during proximity operations, crucial for successful in-space assembly or achieving a precise landing on a surface.

Here are some references to get you started and/or take inspiration from to build the best solution ever!!!:

- [Relative Motion Estimation Based on Sensor Eigenfusion Using a Stereoscopic Vision System and Adaptive Statistical Filtering](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7559174)
- [A 5-Point Minimal Solver for Event Camera Relative Motion Estimation](https://rpg.ifi.uzh.ch/docs/ICCV23_Gao.pdf)
- [Relative Motion Estimation for Vision-based Formation Flight using Unscented Kalman Filter](https://www.researchgate.net/profile/Eric-Johnson-67/publication/250337270_Relative_Motion_Estimation_for_Vision-Based_Formation_Flight_Using_Unscented_Kalman_Filter/links/552290320cf2f9c13052e46a/Relative-Motion-Estimation-for-Vision-Based-Formation-Flight-Using-Unscented-Kalman-Filter.pdf)
- [On The Generation of a Synthetic Event-based Vision Dataset for Navigation and Landing](https://arxiv.org/pdf/2308.00394)
- [Model Based Visual Relative Motion Estimation and Control of a Spacecraft Utilizing Computer Graphics](https://issfd.org/ISSFD_2009/FormationFlyingII/Terui.pdf)


## Project Components 🧩

### 1. Event-Based Camera Setup 🎥🔧

We'll begin by setting up event-based cameras to capture real-time motion events. This involves calibrating the cameras and ensuring they are correctly positioned to monitor the assembly process.
To get ready with the tutorial we can use recorded data for simulation!

### 2. Visual Attention Algorithm Development 🧠💻

Next, we will develop and integrate visual attention algorithms that dynamically shift the focus of attention based on the most relevant parts of the scene (hopefully the aircraft). This step is crucial for reducing data redundancy and enhancing the accuracy of motion estimation.

### 3. Relative Motion Estimation 🛰️🔍

We will implement and test relative motion estimation algorithms to calculate the precise position and orientation of spacecraft components. This involves processing the events captured by the salient areas (visual attention) and using them to determine the relative motion between modules.

### 4. In-Space Assembly Simulation 🔧🌌

Finally, we'll simulate an in-space assembly scenario, using the developed systems to align and dock spacecraft components. This simulation will help validate the effectiveness of our approach and highlight areas for further improvement.

## Getting Started 🚀

1. **Install Required Software**: Ensure you have the necessary software installed (attention.py), including libraries for event-based camera processing and motion estimation.
2. **Run Initial Tests**: Perform initial tests to capture and visualize events throught the visual attention system, verifying that the model is working as expected.
3. **Develop Algorithms**: Start developing the visual attention and motion estimation algorithms, iteratively testing and refining them.
5. **Simulate Assembly**: Use the developed systems to simulate an in-space assembly scenario, analyzing the performance and accuracy of the pose estimation.

## Explanation attention.py  🖥️

This code, developed for the Telluride repository, focuses on converting a video into frames, generating events from these frames, and producing saliency maps for visual attention. It includes several key functions: video2frames extracts frames from a video and saves them; frames2events generates events based on frame differences, considering thresholds for positive and negative changes; see_events visualizes and saves these events; mk_videoframes creates a video from event frames; refactor_data converts event data into a format suitable for the attention model; and run_attention executes the attention model using a spiking CNN with Integrate and Fire neurons, creating saliency maps by processing event frames through layers representing Von Mises filters of different orientations. This model builds on the work by Giulia D’Angelo et al. on event-driven bio-inspired systems for robotics, demonstrating a simplified yet effective approach to visual attention modeling.

## Conclusion 🎉

By the end of this workshop, you'll have a solid understanding of how event-based cameras and visual attention mechanisms can be utilized to enhance in-space assembly operations. This project not only advances the field of space technology but also provides valuable insights into the application of cutting-edge visual processing techniques. We look forward to your contributions and discoveries throughout this exciting journey!

Happy assembling! 🚀✨
