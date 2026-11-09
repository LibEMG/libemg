import os

def get_session_files(path):
    classification = "Classification"
    set1 = ""
    set2 = ""
    set3 = ""
    
    if os.path.isdir(path):
        class_path = os.path.join(path,classification)
        
        for item in os.listdir(class_path):    
            file_path = os.path.join(class_path,item)
                
            if (item == "Classification_Set_1.csv"):
                set1 = file_path
            elif (item == "Classification_Set_2.csv"):
                set2 = file_path
            elif (item == "Classification_Set_3.csv"):
                set3 = file_path
        return {"set1": set1, "set2": set2, "set3": set3}
    else:
        return None