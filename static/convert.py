import os
from pdf2image import convert_from_path

def pdf_to_png(folder_path, output_folder):
    """将指定文件夹中的所有 PDF 文件转换为 PNG 文件"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            images = convert_from_path(pdf_path)

            pdf_name = os.path.splitext(filename)[0]
            for i, image in enumerate(images):
                output_filename = f"{pdf_name}.png"
                output_path = os.path.join(output_folder, output_filename)
                image.save(output_path, "PNG")
                print(f"已保存: {output_path}")

if __name__ == "__main__":
    input_folder = "./image"  # 替换为你的 PDF 文件夹路径
    output_folder = "./image"  # 替换为你的 PNG 输出文件夹路径
    pdf_to_png(input_folder, output_folder)