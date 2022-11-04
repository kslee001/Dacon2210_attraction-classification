__HLN_c3 change log__


1. preprocessing.py
- Expanded coverage for image augmentation 
    - before :  
        - targets = [target_50, target_100, target_300, target_500, target_800]  
    - after :  
        - targets = [target_50, target_100, target_300, target_500, target_800, target_1200, target_1600]  

- Changed method for text augmentation  
    - before :  
        - word tokens random sampling (order of words is not considered)  
    - after :  
        - sentence random sampling + randomly delete word tokens within the sentence (order is now considered)  
    - reason :  
        - to use BERT-based classifier for text classification (KoBertClassifier)  
    
- Added random seed code for all functions  
