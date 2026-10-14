
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. CAU HINH WEBSITE
# =========================
st.set_page_config(
    page_title="Dự báo giá cổ phiếu",
    page_icon="📈",
    layout="wide"
)

st.title("📈 WEBSITE DỰ BÁO GIÁ CỔ PHIẾU")
st.write(
    "Dự báo giá đóng cửa bằng mô hình LSTM "
    "dựa trên dữ liệu lịch sử từ Yahoo Finance."
)

st.warning(
    "Thông tin chỉ phục vụ học tập và nghiên cứu, "
    "không phải khuyến nghị đầu tư."
)

# =========================
# 2. NHAP THONG TIN
# =========================
with st.sidebar:
    st.header("⚙️ Cài đặt dự báo")

    ticker = st.text_input(
        "Mã cổ phiếu",
        value="VNM.VN"
    ).strip().upper()

    start_date = st.date_input(
        "Ngày bắt đầu lấy dữ liệu",
        value=date(2018, 1, 1),
        max_value=date.today() - timedelta(days=60)
    )

    forecast_days = st.slider(
        "Số phiên cần dự báo",
        min_value=1,
        max_value=30,
        value=10
    )

    epochs = st.slider(
        "Số vòng huấn luyện tối đa",
        min_value=5,
        max_value=30,
        value=10
    )

    run_button = st.button(
        "🚀 Chạy dự báo",
        type="primary",
        use_container_width=True
    )

st.caption(
    "Ví dụ mã cổ phiếu Việt Nam: VNM.VN, "
    "FPT.VN, HPG.VN. Khả năng có dữ liệu tùy "
    "mã và nguồn Yahoo Finance."
)

# =========================
# 3. HAM TAI DU LIEU
# =========================
@st.cache_data(ttl=3600)
def download_stock_data(symbol, start):
    end = date.today() + timedelta(days=1)

    df = yf.download(
        symbol,
        start=str(start),
        end=str(end),
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(
            "Không tải được dữ liệu. Hãy kiểm tra mã cổ phiếu "
            "hoặc thử lại sau."
        )

    # Xử lý dữ liệu có nhiều cấp tên cột
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df.xs(
            "Close",
            axis=1,
            level=0
        )
        close = close_data.iloc[:, 0]
    else:
        close = df["Close"]

    result = pd.DataFrame({
        "Date": pd.to_datetime(df.index),
        "Close": pd.to_numeric(
            np.asarray(close).reshape(-1),
            errors="coerce"
        )
    })

    result = result.dropna()
    result = result[result["Close"] > 0]
    result = result.sort_values("Date")

    if len(result) < 100:
        raise ValueError(
            "Dữ liệu có ít hơn 100 phiên. "
            "Hãy chọn mã khác hoặc ngày bắt đầu sớm hơn."
        )

    return result.reset_index(drop=True)


# =========================
# 4. HAM TAO CHUOI 60 PHIEN
# =========================
def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# =========================
# 5. HUAN LUYEN VA DU BAO
# =========================
def run_lstm(df, epochs, forecast_days):
    tf.keras.backend.clear_session()

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    data = df[["Close"]].values.astype("float32")

    # Chia dữ liệu theo thứ tự thời gian
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    if len(train_data) <= 60 or len(test_data) < 1:
        raise ValueError(
            "Không đủ dữ liệu để huấn luyện và kiểm tra."
        )

    # Chỉ học bộ chuẩn hóa từ tập huấn luyện
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled)

    # Ghép 60 phiên cuối của tập train với tập test
    test_combined = np.vstack(
        (train_scaled[-60:], test_scaled)
    )
    X_test, y_test = create_sequences(test_combined)

    X_train = X_train.reshape(
        X_train.shape[0], 60, 1
    )
    X_test = X_test.reshape(
        X_test.shape[0], 60, 1
    )

    # Lấy 10% cuối tập train làm validation
    val_size = max(1, int(0.1 * len(X_train)))

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    if len(X_train_final) == 0:
        raise ValueError("Không đủ dữ liệu huấn luyện.")

    # Kiến trúc LSTM theo code gốc
    model = Sequential([
        tf.keras.Input(shape=(60, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # Dự báo trên tập test
    test_pred_scaled = model.predict(
        X_test,
        verbose=0
    )

    test_pred = scaler.inverse_transform(
        test_pred_scaled
    )

    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    rmse = np.sqrt(
        mean_squared_error(y_test_real, test_pred)
    )
    mae = mean_absolute_error(
        y_test_real, test_pred
    )
    r2 = r2_score(y_test_real, test_pred)

    # Dự báo nối tiếp các phiên tiếp theo
    last_60_days = scaler.transform(data)[-60:]
    last_60_days = last_60_days.reshape(1, 60, 1)

    future_scaled = []

    for _ in range(forecast_days):
        next_pred = model.predict(
            last_60_days,
            verbose=0
        )

        future_scaled.append(next_pred[0, 0])

        last_60_days = np.concatenate(
            [
                last_60_days[:, 1:, :],
                next_pred.reshape(1, 1, 1)
            ],
            axis=1
        )

    future_prices = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

    # Dùng ngày giao dịch tương lai từ lịch sử
    future_dates = pd.bdate_range(
        start=pd.Timestamp(df["Date"].iloc[-1])
              + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        "Phiên dự báo": range(1, forecast_days + 1),
        "Ngày dự báo": future_dates.strftime("%d/%m/%Y"),
        "Giá dự báo": future_prices
    })

    return {
        "model": model,
        "history": history,
        "test_dates": df["Date"].iloc[-len(y_test_real):].reset_index(drop=True),
        "actual": y_test_real.flatten(),
        "predicted": test_pred.flatten(),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "forecast": forecast_df,
        "last_date": df["Date"].iloc[-1],
        "last_price": float(data[-1, 0])
    }


# =========================
# 6. CHAY WEBSITE
# =========================
if run_button:
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if start_date >= date.today():
        st.error("Ngày bắt đầu phải trước ngày hiện tại.")
        st.stop()

    try:
        with st.spinner("Đang tải dữ liệu cổ phiếu..."):
            stock_df = download_stock_data(
                ticker,
                start_date
            )

        st.success(
            f"Đã tải {len(stock_df):,} phiên dữ liệu "
            f"cho mã {ticker}."
        )

        with st.spinner(
            "Mô hình LSTM đang huấn luyện và dự báo. "
            "Lần đầu có thể mất vài phút..."
        ):
            result = run_lstm(
                stock_df,
                epochs,
                forecast_days
            )

        # Thông tin tổng quan
        st.subheader("📊 Tổng quan cổ phiếu")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Giá đóng cửa gần nhất",
            f"{result['last_price']:,.2f}"
        )

        c2.metric(
            "Ngày dữ liệu gần nhất",
            result["last_date"].strftime("%d/%m/%Y")
        )

        c3.metric(
            "Số phiên dự báo",
            f"{forecast_days} phiên"
        )

        # Biểu đồ giá lịch sử
        st.subheader("📈 Giá đóng cửa lịch sử")

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(
            stock_df["Date"],
            stock_df["Close"],
            label="Giá đóng cửa"
        )
        ax1.set_xlabel("Ngày")
        ax1.set_ylabel("Giá")
        ax1.set_title(f"Lịch sử giá cổ phiếu {ticker}")
        ax1.legend()
        ax1.grid(alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # Chỉ số đánh giá
        st.subheader("🧮 Đánh giá mô hình")

        c1, c2, c3 = st.columns(3)

        c1.metric("RMSE", f"{result['rmse']:,.4f}")
        c2.metric("MAE", f"{result['mae']:,.4f}")
        c3.metric("R²", f"{result['r2']:.4f}")

        st.caption(
            "RMSE và MAE càng thấp thường cho thấy sai số "
            "càng nhỏ. R² cần được diễn giải cùng dữ liệu "
            "kiểm tra; không đảm bảo khả năng dự báo tương lai."
        )

        # Biểu đồ giá thật và dự báo trên tập test
        st.subheader("🔍 Giá thực tế và giá mô hình dự báo")

        test_chart = pd.DataFrame({
            "Ngày": result["test_dates"],
            "Giá thực tế": result["actual"],
            "Giá dự báo": result["predicted"]
        }).set_index("Ngày")

        st.line_chart(test_chart)

        # Biểu đồ loss
        st.subheader("📉 Loss trong quá trình huấn luyện")

        loss_df = pd.DataFrame({
            "Train Loss": result["history"].history["loss"],
            "Validation Loss": result["history"].history["val_loss"]
        })

        st.line_chart(loss_df)

        # Dự báo tương lai
        st.subheader(
            f"🔮 Dự báo {forecast_days} phiên tiếp theo"
        )

        st.dataframe(
            result["forecast"].style.format({
                "Giá dự báo": "{:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        fig2, ax2 = plt.subplots(figsize=(12, 5))

        recent = stock_df.tail(60)

        ax2.plot(
            recent["Date"],
            recent["Close"],
            marker="o",
            markersize=3,
            label="60 phiên lịch sử"
        )

        forecast_dates = pd.to_datetime(
            result["forecast"]["Ngày dự báo"],
            format="%d/%m/%Y"
        )

        ax2.plot(
            forecast_dates,
            result["forecast"]["Giá dự báo"],
            marker="o",
            linestyle="--",
            label="Giá dự báo"
        )

        ax2.set_title(
            f"Dự báo giá cổ phiếu {ticker}"
        )
        ax2.set_xlabel("Ngày")
        ax2.set_ylabel("Giá")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # Tải kết quả về máy
        csv_data = result["forecast"].to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "⬇️ Tải bảng dự báo CSV",
            data=csv_data,
            file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )

        st.info(
            "Lưu ý: Ngày dự báo là các ngày làm việc ước tính, "
            "không phải lịch nghỉ lễ chính thức của sàn. "
            "Giá dự báo được tính nối tiếp từ mô hình LSTM."
        )

    except Exception as e:
        st.error(
            "Không thể hoàn thành dự báo. "
            "Hãy kiểm tra mã cổ phiếu, dữ liệu mạng "
            "hoặc thử lại sau."
        )
        st.exception(e)

else:
    st.info(
        "Nhập mã cổ phiếu ở thanh bên trái rồi nhấn "
        "'Chạy dự báo' để bắt đầu."
    )

    st.markdown("""
    ### Hướng dẫn sử dụng
    1. Nhập mã cổ phiếu, ví dụ `VNM.VN`.
    2. Chọn ngày bắt đầu lấy dữ liệu.
    3. Chọn số phiên cần dự báo.
    4. Nhấn **Chạy dự báo**.
    5. Xem biểu đồ, chỉ số đánh giá và tải bảng kết quả.
    """)
