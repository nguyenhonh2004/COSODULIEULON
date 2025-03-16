# Cài đặt và load thư viện
install.packages("sparklyr")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("kableExtra")

library(sparklyr)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(dplyr)  
library(tibble)

# Kết nối Spark
sc <- spark_connect(master = "local")

# Đọc dữ liệu vào Spark DataFrame
data_spark <- spark_read_csv(sc, name = "health_data", 
                             path = "E:/nam3/ki2/dulieulon/Global Health Statistics.csv",
                             header = TRUE, infer_schema = TRUE)
head(data_spark)
str(data_spark)
colnames(data_spark)


# Kiểm tra cấu trúc dữ liệu

data <- as_tibble(data, .name_repair = "unique")

# Chi phí điều trị trung bình theo quốc gia 
average_treatment_cost <- data %>%
  group_by(Country) %>%
  summarise(Average_Treatment_Cost = mean(`Average Treatment Cost (USD)`, na.rm = TRUE))  


# Hiển thị kết quả
print(average_treatment_cost)

# Số bác sĩ trung bình trên 1000 người theo quốc gia
doctors_per_country <- data_spark %>%
  group_by(Country) %>%
  summarise(Average_Doctors = mean(Doctorsper1000, na.rm = TRUE))
print(doctors_per_country )
# Chuyển dữ liệu về R DataFrame để vẽ biểu đồ
doctors_per_country_df <- collect(doctors_per_country)

ggplot(doctors_per_country_df, aes(x = Country, y = Average_Doctors, fill = Country)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Average_Doctors, 2)), vjust = -0.5, size = 4) +
  labs(title = "Bác sĩ trên 1000 người theo quốc gia", x = "Quốc gia", y = "Doctors per 1000") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Tổng hợp các chỉ số y tế theo quốc gia và năm
health_summary <- data_spark %>%
  group_by(Country, Year) %>%
  summarise(
    Avg_MortalityRate = mean(MortalityRate, na.rm = TRUE),
    Avg_HealthcareAccess = mean(HealthcareAccess, na.rm = TRUE),
    Avg_DoctorsPer1000 = mean(Doctorsper1000, na.rm = TRUE),
    Avg_HospitalBedsPer1000 = mean(HospitalBedsper1000, na.rm = TRUE),
    Avg_TreatmentCost = mean(AverageTreatmentCost, na.rm = TRUE),
    Avg_Income = mean(PerCapitaIncome, na.rm = TRUE),
    Avg_EducationIndex = mean(EducationIndex, na.rm = TRUE),
    Avg_UrbanizationRate = mean(UrbanizationRate, na.rm = TRUE)
  )

# Hiển thị kết quả
health_summary_df <- collect(health_summary)
print(health_summary_df, n = 50)  # Hiển thị 50 dòng đầu tiên

# Chọn các cột quan trọng
data_spark <- data_spark %>%
  select(Country, Year, DiseaseName, MortalityRate)

# Chuyển đổi dữ liệu phân loại thành dạng số bằng StringIndexer
data_spark <- data_spark %>%
  ft_string_indexer(input_col = "Country", output_col = "Country_Index") %>%
  ft_string_indexer(input_col = "DiseaseName", output_col = "Disease_Index")


# Hiển thị dữ liệu đã xử lý
glimpse(data_spark)

# Chia tập dữ liệu thành train và test
ml_data <- data_spark %>%  
  sdf_random_split(training = 0.8, testing = 0.2, seed = 1234)

#Hiển thị tập dữ liệu train
glimpse(ml_data$training)
colnames(data_spark)
# Huấn luyện mô hình hồi quy tuyến tính
mortality_model <- ml_linear_regression(ml_data$training, 
                                        response = "MortalityRate",  # Cột cần dự đoán
                                        features = c("Year", "Country_Index", "Disease_Index"))

# Hoặc sử dụng mô hình cây quyết định
mortality_model <- ml_decision_tree_regressor(ml_data$training, 
                                              response = "MortalityRate",
                                              features = c("Year", "Country_Index", "Disease_Index"))

# Dự đoán trên tập kiểm tra
test_predictions <- ml_predict(mortality_model, ml_data$testing)

# Chuyển kết quả về R DataFrame để xem
test_predictions_df <- collect(test_predictions)

# Hiển thị kết quả
print(test_predictions_df, n = 50)  # Hiển thị 50 dòng đầu tiên


ggplot(test_predictions_df %>% filter(DiseaseName == "Ebola"), 
       aes(x = MortalityRate, y = prediction)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(color = "red", linetype = "dashed") +
  labs(title = "So sánh tỷ lệ tử vong thực tế và dự đoán (Ebola)",
       x = "Giá trị thực tế", y = "Giá trị dự đoán") +
  theme_minimal()


# Xác định năm lớn nhất trong tập dữ liệu
max_year <- data_spark %>% summarise(Max_Year = max(Year)) %>% collect() %>% pull(Max_Year)

# Tạo tập dữ liệu dự báo 5 năm tới
future_data <- data_spark %>%
  distinct(Country, DiseaseName) %>%
  collect() %>%  # Chuyển về R DataFrame
  crossing(Year = seq(max_year + 1, max_year + 5))  # Tạo các năm mới


colnames(data_spark)
data_spark %>% select(Country, Country_Index) %>% glimpse()

# Chuyển đổi dữ liệu phân loại
# Chuyển future_data vào Spark
future_data_spark <- sdf_copy_to(sc, future_data, overwrite = TRUE)

# Bây giờ cả hai bảng dữ liệu đều ở Spark => left_join() sẽ hoạt động đúng
future_data_spark <- future_data_spark %>%
  left_join(data_spark %>% select(Country, Country_Index) %>% distinct(), by = "Country") %>%
  left_join(data_spark %>% select(DiseaseName, Disease_Index) %>% distinct(), by = "DiseaseName") %>%
  mutate(Year_Scaled = Year / max(Year, na.rm = TRUE))


# Chuyển vào Spark DataFrame
#future_data_spark <- sdf_copy_to(sc, future_data, overwrite = TRUE)

# Dự đoán tỷ lệ tử vong cho 5 năm tới
future_predictions <- ml_predict(mortality_model, future_data_spark)

# Chuyển về R DataFrame để hiển thị kết quả
future_predictions_df <- collect(future_predictions)

# Hiển thị kết quả
print(future_predictions_df, n = 50)  # Hiển thị 50 dòng kết quả đầu tiên
#print(future_predictions_df, n = Inf)
#View(future_predictions_df)

ggplot(future_predictions_df, aes(x = Year, y = prediction, color = DiseaseName)) +
  geom_line() +
  geom_point() +
  labs(title = "Dự báo tỷ lệ tử vong theo năm", 
       x = "Năm", y = "Tỷ lệ tử vong dự đoán") +
  theme_minimal()

#Xem danh sách quốc gia và bệnbệnh
country_disease_list <- data_spark %>%
  select(Country, DiseaseName) %>%
  distinct() %>%
  collect()

View(country_disease_list)  # Mở cửa sổ xem toàn bộ danh sách

# Chọn bệnh và quốc gia cần xem
selected_disease <- "Ebola"
selected_country <- "USA"

# Lọc dữ liệu
filtered_data <- future_predictions_df %>%
  filter(DiseaseName == selected_disease, Country == selected_country)

# Vẽ biểu đồ
ggplot(filtered_data, aes(x = Year, y = prediction)) +
  geom_line(color = "blue") +
  geom_point(size = 3, color = "red") +
  labs(title = paste("Dự báo tỷ lệ tử vong do", selected_disease, "tại", selected_country),
       x = "Năm", y = "Tỷ lệ tử vong dự đoán") +
  theme_minimal()

# Ngắt kết nối Spark
spark_disconnect(sc)
