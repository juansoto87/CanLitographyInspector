import os
import shutil 


class create_folder:
    def __init__(self):
        self.parent_folder = "/home/juan/Desktop/Pypro/Basics/"
      

    def create_new_folder(self, name):  
        # Directory
        directory = name
        
        # Parent Directory path
        parent_dir = self.parent_folder
        
        # Path
        path = os.path.join(parent_dir, directory)
        self.new_folder = path
        # Create the directory
        # 'GeeksForGeeks' in
        # '/home / User / Documents'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        print("Directory '% s' created" % directory)

    
   

