




    




import libemg

    

    

   
    
    

if __name__ == "__main__":

    libemg.streamers.sifibridge_streamer(version="1_1")


    data_handler = libemg.data_handler.OnlineDataHandler()
    data_handler.start_listening()




    num_reps = 3
    rep_time = 16
    media_folder = "media/"#"images/"#
    data_folder = 'data_regressor/'
    rest_time = 1
    randomize = False
    continuous = False
    gifs = False
    exclude_files = None
    width=800
    height=800
    dc = libemg.screen_guided_training.DataCollection(data_handler)
    dc.launch_training(num_reps,
              rep_time,
              media_folder,
              data_folder,
              rest_time,
              randomize,
              continuous,
              gifs,
              exclude_files,
              width,
              height)
