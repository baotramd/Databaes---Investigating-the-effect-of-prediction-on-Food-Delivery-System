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

def add_pois_to_sample(df_order: pd.DataFrame, poi_ids: list, prediction_date, prediction_hour) -> pd.DataFrame:
    required_pois_df = df_order[df_order['poi_id'].isin(poi_ids)]
    df_sample = pd.DataFrame(columns=required_pois_df.columns)
    for i in poi_ids:
        print(required_pois_df[required_pois_df["poi_id"]==i])
        df_sample = pd.concat([df_sample, required_pois_df[required_pois_df["poi_id"]==i].sample(n=2)])

    print(df_sample)
    return df_sample

if __name__ == '__main__':
    prediction_date = datetime(2022, 10, 24).date()
    interval_num = 2 #current, so forecase num = interval num + 1
    #Assignment rule: assign orders to couriers based on the minimum distance from courier positon to the pickup location
    #file to sample the courier location (for initialization only, as the position should follow the simulation)
    file_courier = "courier_sample.csv"
    #file to sample the pickup location
    file_order = "all_waybill_info_meituan_distinct POI_ID.csv"
    file_result = "Test.xlsx"
    org_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\Python code"
    model_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\VRP_PADTW_MultiCourier"
    sample_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\Sample"
    f1 = os.path.join(org_path, file_courier)
    f2 = os.path.join(org_path, file_order)
    prev_sample = f"Sample_{interval_num-1}.xlsx"
    curr_sample = f"Sample_{interval_num}.xlsx"

    # read old data to array and save it
    arr_result = pd.read_excel(os.path.join(model_path,"Test.xlsx"), sheet_name="Result")
    arr_result["Interval"] = [interval_num - 1 for _ in range(len(arr_result.index))]
    # arr_result[["Interval", "Result"]].to_csv(os.path.join(sample_path,"Result_with pred.csv"), index=False)
    result_total = pd.read_csv(os.path.join(sample_path,"Result_with pred.csv"))
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

    df_cou_node = pd.read_excel(os.path.join(model_path, file_result), sheet_name="courier_node")
    cou_node_mat = df_cou_node.to_numpy()

    startpoint = pd.read_excel(os.path.join(sample_path, prev_sample), sheet_name="Subsequent Initial Position")
    #new order
    poi_ids_to_include = np.array(startpoint["poi_id"].unique())
    poi_ids_to_include = poi_ids_to_include.astype(str)
    df_OrderSample = add_pois_to_sample(df_order, poi_ids_to_include, prediction_date, interval_num)
    print(df_OrderSample)
    #new depot
    df_CourierSample = pd.read_excel(os.path.join(sample_path, prev_sample), sheet_name="Subsequent Initial Position")
    df_CourierSample= df_CourierSample.rename(columns={"poi_id":"courier_id",
                                                       "sender_lng_x":"coord_x", "sender_lat_y":"coord_y"})
    cap = 2
    #assign order
    assignment_matrix = courier_assignment(order_sample=df_OrderSample, courier_sample= df_CourierSample,capacity=cap)
    #generate start points from
    list_startpoint = np.array(df_CourierSample[['courier_id','coord_x', 'coord_y', 'sender prep time']]) # this will be replaced by list_depot of last iteration
    list_pickup = np.array(df_OrderSample[['poi_id','sender_lng_x', 'sender_lat_y', 'sender prep time']].drop_duplicates(subset=["poi_id"], ignore_index=True))
    list_delivery = np.array(df_OrderSample[['Customer number', 'recipient_lng_x', 'recipient_lat_y', 'recipent prep time']]
                             .drop_duplicates(subset=["Customer number"], ignore_index=True))
    node_list = np.block([[list_startpoint], [list_pickup] , [list_delivery]])
    df_NodeList = pd.DataFrame(node_list, columns=["Node_ID", "Coord_x", "Coord_y", "Prep time"]) #coordinates to cal distance matrices
    df_NodeList["Coord_x"] = df_NodeList["Coord_x"].astype(int)
    df_NodeList["Coord_y"] = df_NodeList["Coord_y"].astype(int)
    num_courier = len(df_CourierSample.index)
    pick_mat = pick_matrix(order_sample = df_OrderSample, nodelist = df_NodeList, num_cou=num_courier)
    print(pick_mat)
    deliver_mat = deliver_matrix (order_sample = df_OrderSample, nodelist = df_NodeList, num_cou=num_courier)
    print(deliver_mat)
    cou_node_mat = courier_node(pick = pick_mat, delivery=deliver_mat, assign= assignment_matrix,nodelist=df_NodeList, num_cou=num_courier)
    print("Courier node matrix")
    print(cou_node_mat) # need to rewrite to include prediction and initial points


    #Find the potential pickup point

    pred_pick = pickup_loc_sample(cou_node_mat, df_NodeList, df_data, prediction_date, interval_num + 1)
    pred_pick["dt_hour"] = pd.to_datetime(pred_pick["dt_hour"])
    print(pred_pick)
    df_Depot = predict(prediction_date, interval_num + 1, pred_pick)
    df_Depot = df_Depot.drop_duplicates(subset="2nd_cluster", ignore_index=True).sort_values(by=['2nd_cluster'], ascending=True)
    print(df_Depot)
    df_Depot = df_Depot[['poi_id','sender_lng_x', 'sender_lat_y', 'sender prep time']]
    list_depot = df_Depot.to_numpy()
    node_list = np.block([[node_list],[list_depot]])
    df_NodeList = pd.DataFrame(node_list, columns=["Node_ID", "Coord_x", "Coord_y", "Prep time"]) #coordinates to cal distance matrices
    df_NodeList["Coord_x"] = df_NodeList["Coord_x"].astype(int)
    df_NodeList["Coord_y"] = df_NodeList["Coord_y"].astype(int)
    for i in range(1, num_courier + 1):
        cou_node_mat[-i,-i] = 1


    distance_mat = distMatrix(df_NodeList)

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
    workbook.add_worksheet("Result")

    #write the scalar to the worksheet

    #write the scalar to the worksheet
    ws_scalar.write('A1', 'Number of startpoints')
    ws_scalar.write('B1', list_startpoint.shape[0])
    ws_scalar.write('A2', 'Number of Pickup Points')
    ws_scalar.write('B2', list_pickup.shape[0])
    ws_scalar.write('A3', 'Number of Customers')
    ws_scalar.write('B3', list_delivery.shape[0])
    ws_scalar.write('A4', 'Number of Depot')
    ws_scalar.write('B4', list_depot.shape[0])
    ws_scalar.write('A5', 'Number of Courier')
    ws_scalar.write('B5', len(df_CourierSample.index))
    ws_scalar.write('A6', 'Number of order')
    ws_scalar.write('B6', len(df_OrderSample.index))


    # Write the array to the worksheet.
    for row_num, row_data in enumerate(node_list):
        ws_nodelist.write_row(row_num, 0, row_data)
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

    # save the initial solution

    with pd.ExcelWriter(os.path.join(sample_path, curr_sample)) as writer:
        df_OrderSample.to_excel(writer, sheet_name='Order Sample', index=False)
        df_Depot.to_excel(writer, sheet_name='Depot Sample', index=False)
        df_Depot.to_excel(writer, sheet_name='Subsequent Initial Position', index=False)
        df_CourierSample.to_excel(writer, sheet_name='Initial Position', index=False)

    result_total.to_csv(os.path.join(sample_path,"Result_with pred.csv"), index=False)




