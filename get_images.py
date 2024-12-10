import os
import requests
import mysql.connector


def download_images(image_urls, save_folder):
    # 如果保存图片的文件夹不存在，创建文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, url in enumerate(image_urls):
        try:
            # 发送 GET 请求
            response = requests.get(url)

            # 检查响应状态码是否为 200（请求成功）
            if response.status_code == 200:
                # 从 URL 中提取文件名
                image_name = os.path.join(save_folder, f'image_{i + 1}.jpg')

                # 保存图片到本地文件夹
                with open(image_name, 'wb') as file:
                    file.write(response.content)

                print(f"图片 {i + 1} 已保存: {image_name}")
            else:
                print(f"图片 {i + 1} 请求失败: {url}, 状态码: {response.status_code}")
        except Exception as e:
            print(f"下载图片 {i + 1} 时发生错误: {e}")

# 连接到 MySQL 数据库
conn = mysql.connector.connect(
    host="10.66.52.105",  # 数据库主机地址
    user="root",  # 数据库用户名
    password="zqy5483201",  # 数据库密码
    database="model"  # 数据库名称
)

cursor = conn.cursor()

for i in range(1, 45):
    offset = i * 1000
    sql = "SELECT img_url FROM product_model OFFSET %s LIMIT 1000;"
    cursor.execute(sql, (offset,))
    rows = cursor.fetchall()
    image_urls = [_["img_url"] for _ in rows]
    save_folder = f"downloaded_images_{offset}_{offset + 1000}"
    # 调用函数下载图片
    download_images(image_urls, save_folder)

# 关闭连接
conn.close()
