import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import backend_logic as be # Import file logic

class MLPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Airbnb Price Predictor - NYC")
        self.geometry("800x650")

        self.df = None
        self.param_entries = {}

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Phần 1: Tải và xử lý dữ liệu ---
        data_frame = ttk.LabelFrame(main_frame, text="1. Data Loading & Preprocessing", padding="10")
        data_frame.pack(fill=tk.X, pady=5)

        load_button = ttk.Button(data_frame, text="Load AB_NYC_2019.csv", command=self.load_data)
        load_button.pack(side=tk.LEFT, padx=5)
        self.load_status_label = ttk.Label(data_frame, text="No data loaded.", foreground="red")
        self.load_status_label.pack(side=tk.LEFT, padx=5)

        self.handle_outliers_var = tk.BooleanVar(value=True)
        outlier_check = ttk.Checkbutton(data_frame, text="Handle Outliers (IQR)", variable=self.handle_outliers_var)
        outlier_check.pack(side=tk.LEFT, padx=15)
        
        ttk.Label(data_frame, text="Scaler:").pack(side=tk.LEFT, padx=(10, 2))
        self.scaler_var = tk.StringVar(value="RobustScaler")
        scaler_combo = ttk.Combobox(data_frame, textvariable=self.scaler_var, values=["RobustScaler", "StandardScaler", "MinMaxScaler", "None"])
        scaler_combo.pack(side=tk.LEFT, padx=5)

        # --- Phần 2: Lựa chọn mô hình và tham số ---
        model_frame = ttk.LabelFrame(main_frame, text="2. Model Selection & Hyperparameters", padding="10")
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.model_names = [
            'Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'KNN', 
            'Decision Tree', 'Random Forest', 'Gradient Boosting', 
            'XGBoost', 'LightGBM', 'CatBoost'
        ]
        self.model_var = tk.StringVar(value=self.model_names[-1]) # Default CatBoost
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=self.model_names, width=20)
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind("<<ComboboxSelected>>", self.update_hyperparameter_frame)

        self.hyperparam_frame = ttk.Frame(model_frame, padding="5")
        self.hyperparam_frame.pack(fill=tk.X, pady=10)
        
        # --- Phần 3: Chạy và hiển thị kết quả ---
        action_frame = ttk.LabelFrame(main_frame, text="3. Run & Results", padding="10")
        action_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        run_button = ttk.Button(action_frame, text="Train and Evaluate Model", command=self.run_process)
        run_button.pack(pady=10)

        self.results_text = tk.Text(action_frame, height=12, width=80, state=tk.DISABLED, font=("Courier", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.update_hyperparameter_frame() # Gọi lần đầu để hiển thị tham số cho model mặc định

    def load_data(self):
        # Mở hộp thoại chọn file, nhưng mặc định là file đã biết
        filepath = 'AB_NYC_2019.csv'
        self.df = be.load_data(filepath)
        if self.df is not None:
            self.load_status_label.config(text=f"Loaded {len(self.df)} rows.", foreground="green")
            messagebox.showinfo("Success", f"Successfully loaded {filepath}")
        else:
            self.load_status_label.config(text="File 'AB_NYC_2019.csv' not found!", foreground="red")
            messagebox.showerror("Error", f"Could not find {filepath}. Please place it in the same directory.")

    def update_hyperparameter_frame(self, event=None):
        for widget in self.hyperparam_frame.winfo_children():
            widget.destroy()

        self.param_entries = {}
        model_name = self.model_var.get()
        
        params_to_show = {
            'Ridge': {'alpha': 1.0},
            'Lasso': {'alpha': 1.0},
            'ElasticNet': {'alpha': 1.0, 'l1_ratio': 0.5},
            'KNN': {'n_neighbors': 5},
            'Decision Tree': {'max_depth': 'None', 'min_samples_leaf': 1},
            'Random Forest': {'n_estimators': 100, 'max_depth': 'None'},
            'Gradient Boosting': {'n_estimators': 100, 'learning_rate': 0.1},
            'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1},
            'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1},
            'CatBoost': {'iterations': 100, 'learning_rate': 0.1}
        }

        if model_name in params_to_show:
            for i, (p_name, p_val) in enumerate(params_to_show[model_name].items()):
                ttk.Label(self.hyperparam_frame, text=f"{p_name}:").grid(row=0, column=i*2, padx=5, sticky='w')
                entry = ttk.Entry(self.hyperparam_frame, width=10)
                entry.insert(0, str(p_val))
                entry.grid(row=0, column=i*2+1, padx=5, sticky='w')
                self.param_entries[p_name] = entry

    def get_params_from_gui(self):
        params = {}
        try:
            for p_name, entry in self.param_entries.items():
                val_str = entry.get()
                if p_name in ['n_neighbors', 'n_estimators', 'iterations', 'min_samples_leaf']:
                    params[p_name] = int(val_str)
                elif p_name in ['alpha', 'l1_ratio', 'learning_rate']:
                    params[p_name] = float(val_str)
                elif p_name == 'max_depth':
                    params[p_name] = None if val_str.lower() in ['none', ''] else int(val_str)
            return params
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for hyperparameters.")
            return None

    def run_process(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Processing... Please wait.\n\n")
        self.update_idletasks() # Cập nhật giao diện

        try:
            # 1. Lấy lựa chọn từ GUI
            handle_outliers = self.handle_outliers_var.get()
            scaler_choice = self.scaler_var.get()
            model_name = self.model_var.get()
            params = self.get_params_from_gui()

            if params is None: # Lỗi nhập liệu
                self.results_text.insert(tk.END, "Process cancelled due to invalid input.")
                self.results_text.config(state=tk.DISABLED)
                return

            # 2. Tiền xử lý dữ liệu
            self.results_text.insert(tk.END, f"Preprocessing data (Outliers: {handle_outliers}, Scaler: {scaler_choice})...\n")
            self.update_idletasks()
            X, y = be.preprocess_data(self.df.copy(), handle_outliers, scaler_choice)

            # 3. Huấn luyện và đánh giá
            self.results_text.insert(tk.END, f"Training model: {model_name} with params: {params}...\n\n")
            self.update_idletasks()
            metrics = be.train_and_evaluate(X, y, model_name, params)
            
            # 4. Hiển thị kết quả
            results_str = f"--- Evaluation Metrics for {model_name} ---\n\n"
            results_str += f"{'Metric':<10} | {'Value'}\n"
            results_str += "-"*30 + "\n"
            results_str += f"{'RMSE':<10} | {metrics['RMSE']:.4f}\n"
            results_str += f"{'MAE':<10} | {metrics['MAE']:.4f}\n"
            results_str += f"{'R^2':<10} | {metrics['R^2']:.4f}\n"
            results_str += f"{'MSE':<10} | {metrics['MSE']:.4f}\n"
            results_str += f"{'MAPE (%)':<10} | {metrics['MAPE']:.2f}%\n"

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_str)

        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"An error occurred:\n{str(e)}")
        
        finally:
            self.results_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    app = MLPredictorApp()
    app.mainloop()