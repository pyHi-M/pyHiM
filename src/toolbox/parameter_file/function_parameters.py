import json
import os
import re
from tkinter import messagebox

from core.data_file import save_json


def convert_dic_to_string(dic: dict):
    """Convert a dictionary {'nuclei':'DAPI','mask1':'mask0'} to a simple
    string: 'nuclei: DAPI, mask1: mask0'."""
    string = str(dic).replace("'", "").replace("{", "").replace("}", "")
    return string


def convert_list_to_string(liste: list):
    """Convert a list of type ['masking','clustering'] to a string of type ''."""
    string = str(liste).replace("'", "").replace("[", "").replace("]", "")
    return string


def import_parameters(script_dir, current_dir=None):
    """Function that imports the parameters from the parameters.json
    located in the script directory. If there is a current directory parameter
    the parameters are imported from the parameters.json located in priority from
    the current directory. Return the parameters are stored in a dictionary
    """

    parameters_script_path = script_dir + os.sep + "parameters.json"
    if current_dir is not None:
        parameters_current_path = current_dir + os.sep + "parameters.json"
        if os.path.exists(parameters_current_path):
            with open(parameters_current_path, mode="r") as file:
                dic_full = json.load(file)
        else:
            with open(parameters_script_path, mode="r") as file:
                dic_full = json.load(file)
    else:
        with open(parameters_script_path, mode="r") as file:
            dic_full = json.load(file)

    dic = {}
    dic["dapi_ch"] = dic_full["common"]["acquisition"]["DAPI_channel"]
    dic["dapiFid_ch"] = dic_full["common"]["acquisition"]["fiducialDAPI_channel"]
    dic["barcode_ch"] = dic_full["common"]["acquisition"]["barcode_channel"]
    dic["barcodeFid_ch"] = dic_full["common"]["acquisition"]["fiducialBarcode_channel"]
    dic["mask_ch"] = dic_full["common"]["acquisition"]["mask_channel"]
    dic["maskFid_ch"] = dic_full["common"]["acquisition"]["fiducialMask_channel"]
    dic["rna_ch"] = dic_full["common"]["acquisition"]["RNA_channel"]
    # dic["rnaFid_ch"] = dic_full["common"]["acquisition"][""]
    dic["pixelSizeXY_entry"] = dic_full["common"]["acquisition"]["pixelSizeXY"]
    dic["pixelSizeZ_entry"] = dic_full["common"]["acquisition"]["pixelSizeZ"]
    dic["referenceFiducial_entry"] = dic_full["common"]["alignImages"][
        "referenceFiducial"
    ]
    dic["blockSize_entry"] = dic_full["common"]["alignImages"]["blockSize"]
    dic["flux_min_entry"] = dic_full["common"]["buildsPWDmatrix"]["flux_min"]
    dic["flux_min_3D_entry"] = dic_full["common"]["buildsPWDmatrix"]["flux_min_3D"]
    dic["toleranceDrift_entry"] = dic_full["common"]["buildsPWDmatrix"][
        "toleranceDrift"
    ]
    dic["mask_expansion_entry"] = dic_full["common"]["buildsPWDmatrix"][
        "mask_expansion"
    ]
    dic["folder_entry"] = dic_full["common"]["buildsPWDmatrix"]["folder"]
    dic["masks2process_entry"] = convert_dic_to_string(
        dic_full["common"]["buildsPWDmatrix"]["masks2process"]
    )
    dic["tracing_method_entry"] = convert_list_to_string(
        dic_full["common"]["buildsPWDmatrix"]["tracing_method"]
    )
    dic["KDtree_distance_threshold_mum_entry"] = dic_full["common"]["buildsPWDmatrix"][
        "KDtree_distance_threshold_mum"
    ]
    dic["stardist_entry"] = dic_full["common"]["segmentedObjects"]["stardist_basename"]
    dic["brightest_entry"] = dic_full["common"]["segmentedObjects"]["brightest"]
    dic["aeraMax_dapi_SegObjt_entry"] = dic_full["labels"]["DAPI"]["segmentedObjects"][
        "area_max"
    ]
    dic["aeraMin_dapi_SegObjt_entry"] = dic_full["labels"]["DAPI"]["segmentedObjects"][
        "area_min"
    ]
    dic["zProject_Dapi_zmax_entry"] = dic_full["labels"]["DAPI"]["zProject"]["zmax"]
    dic["zProject_Dapi_zmin_entry"] = dic_full["labels"]["DAPI"]["zProject"]["zmin"]
    dic["zProject_Bcd_zmax_entry"] = dic_full["labels"]["barcode"]["zProject"]["zmax"]
    dic["zProject_Bcd_zmin_entry"] = dic_full["labels"]["barcode"]["zProject"]["zmin"]
    dic["zProject_Mask_zmax_entry"] = dic_full["labels"]["mask"]["zProject"]["zmax"]
    dic["zProject_Mask_zmin_entry"] = dic_full["labels"]["mask"]["zProject"]["zmin"]
    return dic, dic_full


def is_integer(num):
    """Return True if num is an integer, else return False"""
    try:
        convert_num = float(num)
    except Exception:
        return False
    else:
        if convert_num % 1 == 0:
            return True
        else:
            return False


def is_float(num):
    """Return True if num is an integer or float, else return False if string"""
    try:
        float(num)
    except Exception:
        return False
    return True


def check_blocksize(entry_value):
    """Return True if entry_value is a number power of 2, else return False."""
    # to check if a given positive integer is a power of two
    return entry_value > 0 and (entry_value & (entry_value - 1)) == 0


def match_name(reg_expression, string_name):
    """Test string_name with regular expression, return True if the string_name is in the form 'RT+integer'"""
    regex = re.fullmatch(reg_expression, string_name)
    if regex is None:
        return False
    else:
        return True


def check_brightest(entry_value):
    """Convert string in integer, and return the integer, else return the string 'None'
    (=no limit in the number of spot detection)"""
    try:
        output = int(entry_value)
    except Exception:
        output = "None"
    return output


def convert_string_to_dictionnary(string: str) -> dict:
    """Convert a string to a dictionary"""
    dictionary = {}
    temp = string.replace(" ", "").split(",")
    for item in temp:
        list_temp = item.split(":")
        dictionary[list_temp[0]] = list_temp[1]
    return dictionary


def check_dict(string: str):
    """Check if the string correspond to a dictionary, return True if is the case, else return False."""
    try:
        convert_string_to_dictionnary(string)
        return True
    except Exception:
        return False


def update_parameters(user_values_dic, parameters_dic_full, current_dir):
    """ "
    1-Save old parameters from parameters_dic_full in parameters_preversion.json in current directory.
    2-Save new parameters from user_values_dic in parameters.json in current directory.
    """
    dic_comm_acqui = {
        "DAPI_channel": user_values_dic["dapi_ch"],
        "RNA_channel": user_values_dic["rna_ch"],
        "barcode_channel": user_values_dic["barcode_ch"],
        "mask_channel": user_values_dic["mask_ch"],
        "fiducialBarcode_channel": user_values_dic["barcodeFid_ch"],
        "fiducialMask_channel": user_values_dic["maskFid_ch"],
        "fiducialDAPI_channel": user_values_dic["dapiFid_ch"],
        "pixelSizeXY": float(user_values_dic["pixelSizeXY_entry"]),
        "pixelSizeZ": float(user_values_dic["pixelSizeZ_entry"]),
    }
    dic_comm_aligimg = {
        "blockSize": int(user_values_dic["blockSize_entry"]),
        "referenceFiducial": user_values_dic["referenceFiducial_entry"],
    }
    dic_comm_buildmatrix = {
        "tracing_method": user_values_dic["tracing_method_entry"]
        .replace(" ", "")
        .split(","),
        "mask_expansion": int(user_values_dic["mask_expansion_entry"]),
        "flux_min": int(user_values_dic["flux_min_entry"]),
        "flux_min_3D": int(user_values_dic["flux_min_3D_entry"]),
        "KDtree_distance_threshold_mum": int(
            user_values_dic["KDtree_distance_threshold_mum_entry"]
        ),
        "folder": str(user_values_dic["folder_entry"]),
        "masks2process": convert_string_to_dictionnary(
            user_values_dic["masks2process_entry"]
        ),
        "toleranceDrift": int(user_values_dic["toleranceDrift_entry"]),
    }
    dic_comm_segmobj = {
        "stardist_basename": str(user_values_dic["stardist_entry"]),
        "brightest": check_brightest(user_values_dic["brightest_entry"]),
    }
    dic_labels_dapi_segmobj = {
        "area_max": int(user_values_dic["aeraMax_dapi_SegObjt_entry"]),
        "area_min": int(user_values_dic["aeraMin_dapi_SegObjt_entry"]),
    }
    dic_labels_dapi_zpro = {
        "zmax": int(user_values_dic["zProject_Dapi_zmax_entry"]),
        "zmin": int(user_values_dic["zProject_Dapi_zmin_entry"]),
    }
    dic_labels_bcd_zpro = {
        "zmax": int(user_values_dic["zProject_Bcd_zmax_entry"]),
        "zmin": int(user_values_dic["zProject_Bcd_zmin_entry"]),
    }
    dic_labels_mask_zpro = {
        "zmax": int(user_values_dic["zProject_Mask_zmax_entry"]),
        "zmin": int(user_values_dic["zProject_Mask_zmin_entry"]),
    }
    # Save previous parameters contained in parameters_dic in parameters_preversion.json file
    # in the current directory.

    parameters_previous_path = current_dir + os.sep + "parameters_preversion.json"

    save_json(parameters_dic_full, parameters_previous_path)

    # Update parameters_dic with user_values_dic and save new parameters.json
    parameters_dic_full["common"]["acquisition"].update(dic_comm_acqui)
    parameters_dic_full["common"]["alignImages"].update(dic_comm_aligimg)
    parameters_dic_full["common"]["buildsPWDmatrix"].update(dic_comm_buildmatrix)
    parameters_dic_full["common"]["segmentedObjects"].update(dic_comm_segmobj)
    parameters_dic_full["labels"]["DAPI"]["segmentedObjects"].update(
        dic_labels_dapi_segmobj
    )
    parameters_dic_full["labels"]["DAPI"]["zProject"].update(dic_labels_dapi_zpro)
    parameters_dic_full["labels"]["barcode"]["zProject"].update(dic_labels_bcd_zpro)
    parameters_dic_full["labels"]["mask"]["zProject"].update(dic_labels_mask_zpro)

    parameters_new_path = current_dir + os.sep + "parameters.json"
    save_json(parameters_dic_full, parameters_new_path)


def check_settings(entries_dic):
    """Return True if all parameters have good type expected, else return False with a pop-up error window"""
    is_ok = True
    # checks if the inputs correspond to what is expected, else show error window
    for key, list_values in entries_dic.items():
        entered_value = list_values[0].get()
        # test for values that would be normally integer (not string or float)
        if key == "brightest_entry":
            if entered_value != "None" and not is_integer(entered_value):
                messagebox.showerror(
                    "Input Error",
                    f"The type of {key} input is not correct.\nPlease enter integer "
                    f"value or None.",
                )
                is_ok = False
        # test if masks2process_entry is a string that can be converted in a dictionary
        elif key == "masks2process_entry":
            if not check_dict(entered_value):
                messagebox.showerror(
                    "Input Error",
                    f"The type of {key} input is not correct.\nPlease enter string in "
                    f'the form "key1:value1, key2:value2".',
                )
                is_ok = False
        # test if number is an integer and a power of 2
        elif key == "blockSize_entry":
            entered_value = float(entered_value)
            if not check_blocksize(int(entered_value)) or not is_integer(entered_value):
                messagebox.showerror(
                    "Input Error",
                    f"The type of {key} input is not correct.\nPlease enter an integer that is a power of 2.",
                )
                is_ok = False
        # test if name of RT/Barcode is in the form 'RT' + integer
        elif key == "referenceFiducial_entry":
            if not match_name("^RT[0-9][0-9]*", entered_value):
                messagebox.showerror(
                    "Input Error",
                    'The name of Reference Fiducial is not correct.\nPlease enter a name starting with "RT" and followed only by numbers.',
                )
                is_ok = False
        elif list_values[2] is int:
            if not is_integer(entered_value):
                messagebox.showerror(
                    "Input Error",
                    f"The type of {key} input is not correct.\nPlease enter integer value.",
                )
                is_ok = False
        # test for values that would be normally float (not string)
        elif list_values[2] is float:
            if not is_float(entered_value):
                messagebox.showerror(
                    "Input Error",
                    f"The type of {key} input is not correct.\nPlease enter float value.",
                )
                is_ok = False
    return is_ok
