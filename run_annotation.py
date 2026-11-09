from libemg.gui import GUI


if __name__ == "__main__":
    gui = GUI(
        args={
            "annotation_folder": r"B:\DigitShareableData\Aim1\EMG_RawData\Shriner_Participants",
            "sampling_rate": 6000.0,   # whatever the real rate turns out to be
        },
        width=1700,
        height=950,
    )
    gui.start_gui()