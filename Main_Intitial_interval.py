import numpy as np
import pandas as pd
import math
import os
import xlsxwriter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from Prediction import predict
from datetime import datetime

# for the function, please pass the columns representing longitude and latitude to calculate the x, y coord
# each point is defined by poi
def pickup_loc_sample(cou_node_mat, df_NodeList, df_data, pred_date, interval):
    #assign depot node
    assignment_data = []
    centroids = {}
    num_couriers = cou_node_mat.shape[1]

    for courier in range(num_couriers):
        assigned_nodes = np.where(cou_node_mat[:, courier] == 1)[0]
        node_ids = df_NodeList.loc[assigned_nodes, "Node_ID"].values
        coord_x = df_NodeList.loc[assigned_nodes, "Coord_x"].values
        coord_y = df_NodeList.loc[assigned_nodes, "Coord_y"].values
        for node, x, y in zip(node_ids, coord_x, coord_y):
            assignment_data.append({
                "Courier_ID": f"Courier_{courier + 1}",
                "Node_ID": node,
                "Coord_x": x,
                "Coord_y": y
            })

    df_courier_assignments = pd.DataFrame(assignment_data)
    centroids = df_courier_assignments.groupby('Courier_ID').agg({
        'Coord_x': 'mean',
        'Coord_y': 'mean'
    }).reset_index()
    centroids.rename(columns={'Coord_x': 'Centroid_x', 'Coord_y': 'Centroid_y'}, inplace=True)
    centroids = np.array(centroids[["Centroid_x", "Centroid_y"]])
    print(centroids)
    df_data = df_data[df_data['cluster'] == 0]
    points = df_data[['sender_lng_x', 'sender_lat_y']].to_numpy()
    df_data['2nd_cluster'] = kmeans_assignment(centroids, points)
    #sample 100 points from each group
    df_curr_time= df_data[(df_data["dt_hour"].dt.date == pred_date) & (df_data["dt_hour"].dt.hour == interval)]
    uni_pickup_list = df_curr_time[["poi_id", "2nd_cluster"]].drop_duplicates(ignore_index=True)
    pred_pick = uni_pickup_list[uni_pickup_list['2nd_cluster'] == 0].sample(n=50).to_numpy()
    list_cluster = df_data['2nd_cluster'].unique()
    list_cluster.sort()
    for i in list_cluster[1:]:
       pred_pick = np.append(pred_pick,uni_pickup_list[uni_pickup_list['2nd_cluster'] == i].sample(n=50).to_numpy(), axis = 0)
    pred_pick = np.array(pred_pick)
    print(pred_pick.shape)
    pred_pick = pd.DataFrame(pred_pick, columns=["poi_id", "2nd_cluster"])
    pred_pick = pd.merge(pred_pick[["poi_id"]], df_data, on="poi_id", how="left")
    return pred_pick
def pick_matrix(order_sample:pd.DataFrame, nodelist: pd.DataFrame, num_cou)-> np.array:
    pick = [[0 for _ in range(len(order_sample.index))] for _ in range(len(nodelist.index)+num_cou)]
    for i in range(0, len(order_sample.index)):
        ind = np.where(nodelist["Node_ID"] == order_sample.iloc[i]["poi_id"])[0][0]
        pick[ind][i] = 1
    return pick

def deliver_matrix(order_sample:pd.DataFrame, nodelist: pd.DataFrame, num_cou)->np.array:
    delivery = [[0 for _ in range(len(order_sample.index))] for _ in range(len(nodelist.index)+num_cou)]
    for i in range(0, len(order_sample.index)):
        ind = np.where(nodelist["Node_ID"] == order_sample.iloc[i]["Customer number"])[0][0]
        delivery[ind][i] = 1
    return delivery

def courier_node(pick: np.array, delivery: np.array, assign: np.array, nodelist: pd.DataFrame, num_cou)->np.array:
    cour_node = [[0 for _ in range(len(assign[0]))] for _ in range(len(nodelist.index)+num_cou)]
    cour_node = np.array(cour_node)
    node_matrix = np.array(pick) + np.array(delivery)

    for i in range(0, len(assign[0])):
        result = np.matmul(np.array(node_matrix), np.array(assign)[:, i])
        cour_node[:, i] = [1 if x > 0 else 0 for x in result]

    #assign start node to courier
    for i in range(0, len(assign[0])):
        cour_node[i, i] = 1

    # #assign
    # for i in range(1, len(assign[0])+1):
    #     cour_node[-i, -i] = 1
    return cour_node

def courier_assignment(order_sample:pd.DataFrame, courier_sample:pd.DataFrame, capacity:int)->np.array:
    # initialize matrix to store the result
    courier_sample = courier_sample.set_index('courier_id')
    assignment = [[0 for _ in range(len(courier_sample.index))] for _ in range(len(order_sample.index))]
    list_courier = courier_sample.index.to_list()

    #assign for courier to each order
    print("Starting assignment process...")
    for i in range(0, len(order_sample.index)):
        sum_cap = np.array([sum(idx) for idx in zip(*assignment)])
        print(sum_cap)
        # find indices of couriers reaching capacity
        iscap = np.where(sum_cap >= capacity,1,0)
        print(iscap)

        if len(iscap[iscap==1]) < len(courier_sample.index):
            courier = courier_sample[['coord_x','coord_y']].to_numpy()
            print(courier)
            order = order_sample.iloc[i, -3:-1].to_numpy()
            order = np.array([order])
            to_courier = nearest_courier(courier, order, iscap)
            print(f'Result of assignment is: {to_courier[0]}')
            assignment[i][to_courier[0]]=1
        else:
            print('Not all orders are assigned')
            return assignment

    return assignment
def nearest_courier(courier:list, order:list, iscap:list):
    #output df distance between couriers and poi
    num_courier, dim = courier.shape
    num_order, _= order.shape
    print(num_courier, dim)
    print(num_order)
    # Reshape both arrays into `[num_points, num_centroids, dim]`
    courier = np.tile(courier, [num_order, 1]).reshape([num_order, num_courier, dim])
    order = np.tile(order, [1, num_courier]).reshape([num_order, num_courier, dim])

    # Compute all distances (for all points and all centroids) at once and select the min centroid for each point
    distances = np.sum(np.square(courier - order), axis=2)
    distances = np.multiply(distances,np.add([1 for _ in range(num_courier)],[x*100000 for x in iscap]))
    print(distances)

    print(np.argmin(distances, axis=1))
    return np.argmin(distances, axis=1)

def kmeans_assignment(centroids, points):
    #fixed centroid from clustering
    # centroids = np.array([[1941101.23058443, 575383.85270368],
    #                      [1945283.84547307, 578199.51849496]])
    num_centroids, dim = centroids.shape
    num_points, _ = points.shape

    # Reshape both arrays into `[num_points, num_centroids, dim]`
    centroids = np.tile(centroids, [num_points, 1]).reshape([num_points, num_centroids, dim])
    points = np.tile(points, [1, num_centroids]).reshape([num_points, num_centroids, dim])

    # Compute all distances (for all points and all centroids) at once and select the min centroid for each point
    distances = np.sum(np.square(centroids - points), axis=2)
    print(distances)
    return np.argmin(distances, axis=1)

def coor2cartesian(df: pd.DataFrame, lng: str, lat: str) -> pd.DataFrame:
    # convert longitude and latitude to x, y
    #pass lng column and lat column to the function to calculate x and y
    R = 6371  # radius of the Earth in kilometers
    lon0 = 0  # central meridian
    df[lng + '_downscale'] = df[lng] / 1000000.00
    df[lat + '_downscale'] = df[lat] / 1000000.00
    df[lng + '_x'] = df.apply(lambda x: 100 * R * math.radians(x[lng + '_downscale'] - lon0), axis=1)
    df[lat + '_y'] = df.apply(lambda x: 100 * R * math.log(math.tan(math.pi / 4 + math.radians(x[lat + '_downscale']) / 2)),
                             axis=1)
    df.drop(columns=[lng + '_downscale', lat + '_downscale'], inplace= True)
    return df

# for the function, the column "sender_x" and "sender_y" are used to calculate
# each point is defined by poi
def distMatrix(df: pd.DataFrame) -> np.array:
    # dist_matrix = pd.DataFrame(squareform(pdist(df.loc[:, ["Coord_x","Coord_y"]])), columns=df["Node_ID"].unique(),
    #                            index=df["Node_ID"].unique())
    dist_matrix = np.array(squareform(pdist(df.loc[:, ["Coord_x","Coord_y"]])))/32.55
    return dist_matrix

if __name__ == '__main__':
    prediction_date = datetime(2022, 10, 24).date()
    interval_num = 0
    #Assignment rule: assign orders to couriers based on the minimum distance from courier positon to the pickup location
    #file to sample the courier location (for initialization only, as the position should follow the simulation)
    file_courier = "courier_sample.csv"
    #file to sample the pickup location
    file_order = "all_waybill_info_meituan_distinct POI_ID.csv"
    # poi_preptime = "meal prep time.csv"
    org_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\Python code"
    model_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\VRP_PADTW_MultiCourier"
    sample_path = "C:\\Users\\baotr\\OneDrive\\Documents\\Study\\AI Seminar\\Sample"
    f1 = os.path.join(org_path, file_courier)
    f2 = os.path.join(org_path, file_order)
    sample_file = f"Sample_{interval_num}.xlsx"
    excel_file = 'Test.xlsx'


    df_courier = pd.read_csv(f1)
    df_courier["prep time"] = [0 for _ in range(len(df_courier.index))]
    df_order = pd.read_csv(f2)
    # df_preptime = pd.read_csv(f3)
    df_order['dt_hour'] = pd.to_datetime(df_order['dt_hour'])

    # df_order = df_order.merge(df_preptime,on = "poi_id", how = "left")
    df_order['recipent prep time'] = [0 for _ in range(len(df_order.index))]
    df_order = df_order.sort_values(by=['order_time'], ascending=True)
    df_data = pd.read_csv(os.path.join(org_path, "Pickup Location Demand by time.csv"))
    df_data["dt_hour"] = pd.to_datetime(df_data["dt_hour"])

    # df_order = coor2cartesian(df_order, 'sender_lng', 'sender_lat')
    # df_order = coor2cartesian(df_order, 'recipient_lng', 'recipient_lat')
    # df_courier = coor2cartesian(df_courier, 'rider_lng', 'rider_lat')

    # # demand data generation
    # df_data = (df_order[["dt_hour",'poi_id', 'sender_lng_x', 'sender_lat_y', "sender prep time", "cluster"]].
    #       value_counts(["dt_hour","poi_id"]))
    # df_data = (df_order[["dt_hour",'poi_id', 'sender_lng_x', 'sender_lat_y', "sender prep time", "cluster"]].drop_duplicates(ignore_index=True)
    #       .join(df_data, on=['dt_hour','poi_id']))
    # unq_pickup = df_data[['poi_id', 'sender_lng_x', 'sender_lat_y', "sender prep time", "cluster"]].drop_duplicates(ignore_index=True)
    #
    # complete_hours = pd.DataFrame({'order_hour': range(24)})
    # # Create a unique combination of poi_id and date
    # df_data['dt'] = pd.to_datetime(df_data['dt_hour'].dt.date, format='%d/%m/%Y')
    # unique_pois_dates = df_data[['poi_id', 'dt']].drop_duplicates()
    #
    # cartesian = unique_pois_dates.merge(complete_hours, how='cross')
    # cartesian["dt_hour"] = pd.to_datetime(cartesian['dt'].astype(str) + ' ' + cartesian['order_hour'].astype(str) + ':00:00')
    # filled_df = cartesian.merge(df_data, on=["dt_hour", 'poi_id'], how='left')
    # filled_df['count'] = filled_df['count'].fillna(0).astype(int)
    #
    # dtypes = {
    #     'dt_hour': 'datetime64[ns]',
    #     'poi_id': 'object',
    #     'sender_lng_x': 'float64',
    #     'sender_lat_y': 'float64',
    #     "sender prep time": 'float64',
    #     'count': 'int64',
    #     "cluster":'int64'
    # }
    #
    # final_df = filled_df[["dt_hour",'poi_id', 'count']]
    #
    # final_df=pd.merge(final_df, unq_pickup, on = "poi_id", how = "left")
    # final_df.reset_index(drop=True,inplace = True)
    #
    # if os.path.exists(os.path.join(org_path, "Pickup Location Demand by time.csv")):
    #     os.remove(os.path.join(org_path, "Pickup Location Demand by time.csv"))
    # final_df.to_csv(os.path.join(org_path, "Pickup Location Demand by time.csv"), index=False)


    # cluster the points using knn, uncommented out for the first time only to find the cluster
    # X = df_order[['sender_lng_x', 'sender_lat_y']]
    # fixed_center = [[1941126.43803898, 575000.04746468],
    #                 [1945287.1790246, 578199.38839885]]
    # kmeans = KMeans(n_clusters=2, init=np.array(fixed_center), n_init=1)
    # kmeans.fit(X)  # Compute k-means clustering.
    # df_order['cluster'] = kmeans.fit_predict(X)
    # if os.path.exists(f2):
    #     os.remove(f2)
    # df_order.to_csv(f2, index=False)

    ## assign couriers to clusters using Kmeans
    # centroid = np.array(kmeans.cluster_centers_)
    # print(centroid)

    # # assign cluster to couriers, uncommented out for the first time only to assign the cluster
    # # fixed centroid from clustering

    # # centroids = np.array([[1941101.23058443, 575383.85270368],
    # #             [1945283.84547307, 578199.51849496]])
    # points = df_courier[['coord_x', 'coord_y']].to_numpy()
    # print(points.shape)
    # df_courier['cluster'] = kmeans_assignment(points)
    # plt.scatter(df_courier['coord_x'], df_courier['coord_y'], s=df_courier['courier_id'], c=df_courier['cluster'])
    # plt.show()
    # if os.path.exists(f1):
    #     os.remove(f1)
    # df_courier.to_csv(f1, index=False)


    # #only take from cluster 1 for both orders and couriers and sample randomly
    # #the sampling is only run once as different results are sampled each time the code run. The result should be saved to a file
    # df_OrderSample = df_order[df_order['cluster']==0].sample(n = 10)
    # df_Depot = df_order[df_order['cluster'] == 0].sample(n=5)
    # df_CourierSample = df_courier[df_courier['cluster'] == 0].sample(n = 5)


    df_OrderSample = pd.read_excel(os.path.join(sample_path, sample_file), sheet_name="Order Sample")
    df_CourierSample = pd.read_excel(os.path.join(sample_path, sample_file), sheet_name="Initial Position")
    cap = 2
    #assign order
    assignment_matrix = courier_assignment(order_sample=df_OrderSample, courier_sample= df_CourierSample,capacity=cap)
    #generate start points from
    list_startpoint = np.array(df_CourierSample[['courier_id','coord_x', 'coord_y', "prep time"]]) # this will be replaced by list_depot of last iteration
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
    ws_result.write('A1', 'Result')

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
    with pd.ExcelWriter(os.path.join(sample_path, sample_file)) as writer:
        df_OrderSample.to_excel(writer, sheet_name='Order Sample')
        df_Depot.to_excel(writer, sheet_name='Depot Sample')
        df_Depot.to_excel(writer, sheet_name='Subsequent Initial Position')
        df_CourierSample.to_excel(writer, sheet_name='Initial Position')





