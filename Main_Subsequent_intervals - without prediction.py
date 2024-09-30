import numpy as np
import pandas as pd
import math
import os
import xlsxwriter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from Main_Intitial_interval import pick_matrix, deliver_matrix, courier_node, courier_assignment, nearest_courier
from Main_Intitial_interval import kmeans_assignment, coor2cartesian, distMatrix, pickup_loc_sample
from Prediction import predict
from datetime import datetime


if __name__ == '__main__':
    prediction_date = datetime(2022, 10, 24).date()
    interval_num = 1 #current, so forecase num = interval num + 1
    #Assignment rule: assign orders to couriers based on the minimum distance from courier positon to the pickup location
    #file to sample the courier location (for initialization only, as the position should follow the simulation)
    file_courier = "courier_sample.csv"
    #file to sample the pickup location
    file_order = "all_waybill_info_meituan_distinct POI_ID.csv"
    file_result = "Test.xlsx"
    org_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\Python code"
    model_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\VRP_PADTW_MultiCourier_without prediction"
    sample_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\Sample"
    f1 = os.path.join(org_path, file_courier)
    f2 = os.path.join(org_path, file_order)
    curr_sample = f"Sample_{interval_num}.xlsx"

    # read old data to array and save it
    arr_result = pd.read_excel(os.path.join(model_path,"Test.xlsx"), sheet_name="Result")
    arr_result["Interval"] = [interval_num - 1 for _ in range(len(arr_result.index))]
    # arr_result[["Interval", "Result"]].to_csv(os.path.join(sample_path,"Result_without pred.csv"), index=False)
    result_total = pd.read_csv(os.path.join(sample_path,"Result_without pred.csv"))
    if interval_num - 1 not in result_total["Interval"].unique():
        result_total = pd.concat([result_total,arr_result])


    # df_courier = pd.read_csv(f1)
    # df_courier["prep time"] = [0 for _ in range(len(df_courier.index))]
    df_order = pd.read_csv(f2)
    # df_preptime = pd.read_csv(f3)
    df_order['dt_hour'] = pd.to_datetime(df_order['dt_hour'])

    # df_order = df_order.merge(df_preptime,on = "poi_id", how = "left")
    df_order = df_order.sort_values(by=['dt_hour'], ascending=True)
    df_data = pd.read_csv(os.path.join(org_path, "Pickup Location Demand by time.csv"))
    df_data["dt_hour"] = pd.to_datetime(df_data["dt_hour"])

    df_OrderSample = pd.read_excel(os.path.join(sample_path, curr_sample), sheet_name="Order Sample")
    df_Depot = pd.read_excel(os.path.join(sample_path, curr_sample), sheet_name="Depot Sample")
    df_NodeList = pd.read_excel(os.path.join(model_path, "Test.xlsx"), sheet_name="Node list")
    visit_time = pd.read_excel(os.path.join(model_path, "Test.xlsx"), sheet_name="Visit time").to_numpy()
    endpoint = np.argmax(visit_time, axis=0)
    df_CourierSample = df_NodeList[df_NodeList.index == endpoint[0]]
    for i in range(1, len(endpoint)):
        df_CourierSample = pd.concat([df_CourierSample,df_NodeList[df_NodeList.index == endpoint[i]]])
    df_CourierSample= df_CourierSample.rename(columns={"Unnamed: 0":"courier_id",
                                                       "Unnamed: 1":"coord_x", "Unnamed: 2":"coord_y",
                                                       "Unnamed: 3":"prep time"})
    print(df_CourierSample)
    print(df_OrderSample.iloc[i, -3:-1].to_numpy())
    cap = 2
    # assign order
    assignment_matrix = courier_assignment(order_sample=df_OrderSample, courier_sample=df_CourierSample, capacity=cap)
    # generate start points from
    list_startpoint = np.array(df_CourierSample[['courier_id', 'coord_x', 'coord_y',
                                                 "prep time"]])  # this will be replaced by list_depot of last iteration
    list_pickup = np.array(
        df_OrderSample[['poi_id', 'sender_lng_x', 'sender_lat_y', 'sender prep time']].drop_duplicates(
            subset=["poi_id"], ignore_index=True))
    list_delivery = np.array(
        df_OrderSample[['Customer number', 'recipient_lng_x', 'recipient_lat_y', 'recipent prep time']]
        .drop_duplicates(subset=["Customer number"], ignore_index=True))
    node_list = np.block([[list_startpoint], [list_pickup], [list_delivery]])
    df_NodeList = pd.DataFrame(node_list, columns=["Node_ID", "Coord_x", "Coord_y",
                                                   "Prep time"])  # coordinates to cal distance matrices
    df_NodeList["Coord_x"] = df_NodeList["Coord_x"].astype(int)
    df_NodeList["Coord_y"] = df_NodeList["Coord_y"].astype(int)
    num_courier = 0  # no depot
    distance_mat = distMatrix(df_NodeList)
    pick_mat = pick_matrix(order_sample=df_OrderSample, nodelist=df_NodeList, num_cou=num_courier)
    print(pick_mat)
    deliver_mat = deliver_matrix(order_sample=df_OrderSample, nodelist=df_NodeList, num_cou=num_courier)
    print(deliver_mat)
    cou_node_mat = courier_node(pick=pick_mat, delivery=deliver_mat, assign=assignment_matrix, nodelist=df_NodeList,
                                num_cou=num_courier)
    #write to excel sheets
    # Create a workbook and add a worksheet.

    excel_file = 'Test.xlsx'
    if os.path.exists(os.path.join(model_path,excel_file)):
        os.remove(os.path.join(model_path,excel_file))
    workbook = xlsxwriter.Workbook(os.path.join(model_path,excel_file))
    ws_scalar = workbook.add_worksheet("Scalar")
    ws_nodelist = workbook.add_worksheet("Node list")
    ws_order_assignment = workbook.add_worksheet("Assignment matrix")
    ws_order_pick = workbook.add_worksheet("Order pick")
    ws_order_deliver = workbook.add_worksheet("Order delivery")
    ws_courier_node = workbook.add_worksheet("courier_node")
    ws_time = workbook.add_worksheet("Distance and Time")
    ws_result = workbook.add_worksheet("Result")
    workbook.add_worksheet("Visit time")

    #write the scalar to the worksheet

    #write the scalar to the worksheet
    ws_scalar.write('A1', 'Number of startpoints')
    ws_scalar.write('B1', list_startpoint.shape[0])
    ws_scalar.write('A2', 'Number of Pickup Points')
    ws_scalar.write('B2', list_pickup.shape[0])
    ws_scalar.write('A3', 'Number of Customers')
    ws_scalar.write('B3', list_delivery.shape[0])
    ws_scalar.write('A4', 'Number of Courier')
    ws_scalar.write('B4', len(df_CourierSample.index))
    ws_scalar.write('A5', 'Number of order')
    ws_scalar.write('B5', len(df_OrderSample.index))
    ws_result.write('A1',  "Result")

    # Write the array to the worksheet.
    for row_num, row_data in enumerate(node_list):
        ws_nodelist.write_row(row_num + 1, 0, row_data)
    for row_num, row_data in enumerate(assignment_matrix):
        ws_order_assignment.write_row(row_num, 0, row_data)
    for row_num, row_data in enumerate(pick_mat):
        ws_order_pick.write_row(row_num, 0, row_data)
    for row_num, row_data in enumerate(deliver_mat):
        ws_order_deliver.write_row(row_num, 0, row_data)
    for row_num, row_data in enumerate(cou_node_mat):
        ws_courier_node.write_row(row_num, 0, row_data)
    for row_num, row_data in enumerate(distance_mat):
        ws_time.write_row(row_num, 0, row_data)


    # Close the workbook.
    workbook.close()

    # save the result
    result_total.to_csv(os.path.join(sample_path,"Result_without pred.csv"), index=False)




