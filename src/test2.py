import pandas as pd

data = [
    [1, "Vietcombank (VCB)", "VN_VCB ESG 2020_250519.pdf", 2020, "Ngân hàng", "Thường niên (Tích hợp)"],
    [2, "Vietcombank (VCB)", "VN_VCB ESG 2023_250519.pdf", 2023, "Ngân hàng", "ESG"],
    [3, "Vietcombank (VCB)", "VN_VCB ESG 2024_250519.pdf", 2024, "Ngân hàng", "ESG"],
    [4, "Ngân hàng Á Châu (ACB)", "ACB_ESG_2022.pdf", 2022, "Ngân hàng", "ESG"],
    [5, "Ngân hàng Á Châu (ACB)", "ACB_ESG_2023.pdf", 2023, "Ngân hàng", "ESG"],
    [6, "Ngân hàng Á Châu (ACB)", "ACB_ESG_2024.pdf", 2024, "Ngân hàng", "ESG"],
    [7, "Techcombank (TCB)", "TCB_2024.pdf", 2024, "Ngân hàng", "ESG"],
    [8, "BIDV", "BIDV_BC+PTBV_2023.pdf", 2023, "Ngân hàng", "ESG"],
    [9, "BIDV", "BIDV_BC+PTBV_2024.pdf", 2024, "Ngân hàng", "ESG"],
    [10, "Tập đoàn FPT", "BCTN FPT 2020 VN.pdf", 2020, "Công nghệ", "Thường niên (Tích hợp)"],
    [11, "Tập đoàn FPT", "BCTNFPT 2022.pdf", 2022, "Công nghệ", "Thường niên (Tích hợp)"],
    [12, "Tập đoàn FPT", "BCTN FPT 2023 VN.pdf", 2023, "Công nghệ", "Thường niên (Tích hợp)"],
    [13, "Tập đoàn FPT", "BCTN FPT 2024 VN.pdf", 2024, "Công nghệ", "Thường niên (Tích hợp)"],
    [14, "Vinamilk", "BCTN Vinamil 2022 VN.pdf", 2022, "Hàng tiêu dùng (F&B)", "ESG"],
    [15, "Vinamilk", "BCTN Vinamil 2023 VN.pdf", 2023, "Hàng tiêu dùng (F&B)", "ESG"],
    [16, "Vinamilk", "BCTN Vinamil 2024 VN.pdf", 2024, "Hàng tiêu dùng (F&B)", "ESG"],
    [17, "PV Trans", "BCTN - PV Trans.pdf", 2024, "Vận tải & Logistics", "Thường niên (Tích hợp)"],
    [18, "Vietourist Holdings", "VI_BCTN 2022.pdf", 2022, "Du lịch & Dịch vụ", "Thường niên"],
    [19, "Vietourist Holdings", "BCTN 2023.pdf", 2023, "Du lịch & Dịch vụ", "Thường niên"],
    [20, "Vietourist Holdings", "VI_BCTN 2024.pdf", 2024, "Du lịch & Dịch vụ", "Thường niên"],
]

df = pd.DataFrame(data, columns=["TT", "Tên Doanh nghiệp", "Tên Báo cáo", "Năm", "Lĩnh vực Hoạt động", "Loại Báo cáo"])
df.to_excel("Danh_sach_bao_cao_ESG.xlsx", index=False)