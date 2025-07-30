import json
import random

industries = [
    "năng lượng", "dệt may", "thực phẩm", "tài chính", "bất động sản",
    "vận tải", "nông nghiệp", "dược phẩm", "viễn thông", "sản xuất ô tô"
]

locations = ["Bình Dương", "Đồng Nai", "Hải Phòng", "Cần Thơ", "Quảng Ngãi"]
regions = ["miền Tây", "miền Trung", "miền Bắc", "TP.HCM", "Tây Nguyên"]

extended_templates = {
    "environmental": [
        "Nhà máy tại {location} đã giảm lượng phát thải CO₂ xuống còn {value} tấn/năm.",
        "Tại vùng {region}, doanh nghiệp đầu tư {amount} vào hệ thống điện mặt trời áp mái.",
        "Chúng tôi triển khai hệ thống tuần hoàn nước tại khu công nghiệp {location}, giúp tái sử dụng {percent} lượng nước thải.",
        "Lượng tiêu thụ điện bình quân giảm còn {kwh} kWh/người/tháng trong toàn hệ thống logistics.",
        "Cơ sở sản xuất đã thay thế 100% nhiên liệu hóa thạch bằng năng lượng sinh khối từ năm {year}.",
        "Dự án nông nghiệp hữu cơ giúp giảm {percent} lượng thuốc bảo vệ thực vật ra môi trường."
    ],
    "social": [
        "Tỷ lệ nhân viên nữ tham gia chương trình đào tạo kỹ thuật tại khu vực {region} đạt {percent}.",
        "Nhân viên ngành dệt may tại {location} được hỗ trợ nhà ở với kinh phí lên đến {amount}.",
        "Chúng tôi đã hợp tác với bệnh viện địa phương để cung cấp khám sức khỏe định kỳ cho hơn {number} công nhân.",
        "Tổ chức {number} buổi đối thoại cộng đồng tại huyện {location} trong năm {year}.",
        "Chương trình ‘An toàn cho mọi người’ đã giảm {percent} tai nạn lao động tại các công trường."
    ],
    "governance": [
        "Ban giám đốc tại chi nhánh {region} có {percent} là thành viên độc lập.",
        "Từ năm {year}, công ty áp dụng hệ thống đánh giá rủi ro ESG định kỳ theo quý.",
        "Chúng tôi yêu cầu tất cả nhà cung cấp tại khu vực {region} ký cam kết về đạo đức kinh doanh.",
        "100% hợp đồng có điều khoản bắt buộc tuân thủ các tiêu chuẩn môi trường nội bộ.",
        "Các chính sách về chống rửa tiền được cập nhật theo quy định tại thị trường {region}."
    ]
}

exaggerated_templates = [
    "Chúng tôi là công ty tiên phong và bền vững nhất trong ngành {industry}.",
    "Không có doanh nghiệp nào đạt được thành tựu ESG vượt trội như chúng tôi.",
    "Chúng tôi cam kết đảm bảo an toàn tuyệt đối cho toàn bộ người lao động.",
    "Công ty luôn đạt được sự tin tưởng tuyệt đối từ cộng đồng và đối tác.",
    "Không có vi phạm nào từng xảy ra trong hệ thống chuỗi cung ứng của chúng tôi.",
    "Chúng tôi luôn đi đầu trong mọi tiêu chuẩn ESG trên thị trường."
]

exaggerated_reasons = [
    "Câu khẳng định tuyệt đối mà không có dữ liệu cụ thể để kiểm chứng.",
    "Sử dụng từ như 'luôn', 'tuyệt đối', 'không bao giờ' là dấu hiệu phóng đại.",
    "Không có bằng chứng hỗ trợ cho tuyên bố mang tính chủ quan và toàn diện."
]

def generate_exaggerated_sentence(industry):
    sentence = random.choice(exaggerated_templates).format(industry=industry)
    reason = random.choice(exaggerated_reasons)
    return sentence, reason

def generate_diverse_factual_sentence():
    category = random.choice(list(extended_templates.keys()))
    template = random.choice(extended_templates[category])
    return template.format(
        location=random.choice(locations),
        region=random.choice(regions),
        year=random.choice(["2020", "2021", "2022", "2023", "2024"]),
        percent=f"{random.randint(15, 95)}%",
        value=f"{random.randint(100, 10000)}",
        kwh=f"{random.randint(120, 350)}",
        amount=f"{random.randint(2, 100)} tỷ VNĐ",
        number=random.randint(2, 50)
    )

def generate_esg_dataset(num_samples=50, allow_exaggeration=True, save_path="esg_testset.json"):
    used_sentences = set()
    all_samples = []

    while len(all_samples) < num_samples:
        industry = random.choice(industries)
        force_no_exag = random.random() < 0.25
        num_exaggerated = 0 if force_no_exag else random.randint(1, 4)
        num_factual = 10 - num_exaggerated

        exaggerated = []
        factual = []

        retry = 0
        while len(exaggerated) < num_exaggerated and retry < 50:
            s, r = generate_exaggerated_sentence(industry)
            if s not in used_sentences:
                exaggerated.append((s, r))
                used_sentences.add(s)
            retry += 1

        retry = 0
        while len(factual) < num_factual and retry < 100:
            s = generate_diverse_factual_sentence()
            if s not in used_sentences:
                factual.append(s)
                used_sentences.add(s)
            retry += 1

        if len(exaggerated) + len(factual) == 10:
            all_sentences = [e[0] for e in exaggerated] + factual
            random.shuffle(all_sentences)
            output = [{"sentence": e[0], "reason": e[1]} for e in exaggerated]
            sample = {
                "instruction": f"Đoạn văn sau được trích từ báo cáo ESG của một công ty trong ngành {industry}. Hãy xác định các câu có dấu hiệu phóng đại và giải thích lý do.",
                "input": all_sentences,
                "output": {
                    "label": "phóng đại" if output else "không phóng đại",
                    "sentences": output
                }
            }
            all_samples.append(sample)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã tạo {num_samples} mẫu ESG thành công tại: {save_path}")

# Ví dụ sử dụng:
if __name__ == "__main__":
    generate_esg_dataset(num_samples=50, save_path="testset_esg_50_diverse.json")
