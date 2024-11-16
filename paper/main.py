from tool_Class import *


def df_select(dataframe, time, is_Service_Provider):
    return dataframe[(dataframe['time'] == round(time, 1)) & (dataframe['is_Service_Provider'] == is_Service_Provider)]


def print_all(is_Service_Provider=False, total_time=600):
    start = 0.0
    for _ in range(total_time * 10 + 1):
        print("==========================================================================")
        print(df_select(df, start, is_Service_Provider))
        print("==========================================================================")
        print("\n" * 5)
        start += 0.1


if __name__ == '__main__':

    df = Simulation().run()

    df = df.sort_values(by='time').reset_index(drop=True)

    # print_all(total_time=10)

    print(df.head(50))


    ## 保存到文件
    df.to_csv('data.csv', index=True)
    df.to_excel('data.xlsx', index=True)
