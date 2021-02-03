import os.path as path


# GLOBALS
DIR_WORKDAY=r"/Users/alexker/Desktop/workspace/neurable/spring/"
WorkdayFp=lambda rel_pth: path.join(DIR_WORKDAY, rel_pth)

# Raw data files --
DIR_EEG=WorkdayFp("workday_datasets")
DIR_LBL=WorkdayFp("workday_datasets")


if __name__=="__main__":
    # Confirm that configured directories exists
    assert path.isdir(DIR_EEG)
    assert path.isdir(DIR_LBL)