import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the CSV file
file_path = "frame_1746803997.279s.csv"

class DataFormatConverter:
    def __init__(self):
        image_pattern = [
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
                ]
        lookup_table = [
                [16,   4, 10,  5,  0,  1, -1, -1, -1, -1],
                [32,  24, 17, 11,  6,  2, -1, -1, -1, -1],
                [41,  33, 25, 18, 12,  7,  3, -1, -1, -1],
                [59,  50, 42, 34, 26, 19, 13,  8, -1, -1],
                [70,  60, 51, 43, 35, 27, 20, 14,  9, -1],
                [81,  71, 61, 52, 44, 36, 28, 21, 15, -1],
                [91,  82, 72, 62, 53, 45, 37, 29, 22, -1],
                [101, 92, 83, 73, 63, 54, 46, 38, 30, 23],
                [111,102, 93, 84, 74, 64, 55, 47, 39, 31],
                [121,112,103, 94, 85, 75, 65, 56, 48, 40],
                [130,122,113,104, 95, 86, 76, 66, 57, 49],
                [139,131,123,114,105, 96, 87, 77, 67, 58],
                [140,132,124,115,106, 97, 88, 78, 68, -1],
                [147,141,133,125,116,107, 98, 89, 79, 69],
                [154,148,142,134,126,117,108, 99, 90, 80],
                [161,155,149,143,135,127,118,109,100, -1],
                [162,156,150,144,136,128,119,110, -1, -1],
                [168,163,157,151,145,137,129,120, -1, -1],
                [174,169,164,158,152,146,138, -1, -1, -1],
                [180,175,170,165,159,153, -1, -1, -1, -1],
                [192,186,181,176,171,166,160, -1, -1, -1],
                [199,193,187,182,177,172,167, -1, -1, -1],
                [200,194,188,183,178,173, -1, -1, -1, -1],
                [207,201,195,189,184,179, -1, -1, -1, -1],
                [214,208,202,196,190,185, -1, -1, -1, -1],
                [221,215,209,203,197,191, -1, -1, -1, -1],
                [228,222,216,210,204,198, -1, -1, -1, -1],
                [235,229,223,217,211,205, -1, -1, -1, -1],
                [236,230,224,218,212,206, -1, -1, -1, -1],
                [242,237,231,225,219,213, -1, -1, -1, -1],
                [248,243,238,232,226,220,227,234,241,247],
                [249,244,250,251,245,239,233,240,246,252]
                ]
        self.image_pattern_array = np.array(image_pattern)
        self.lookup_table_array = np.array(lookup_table)
        lookup_table_rows, lookup_table_cols = self.lookup_table_array.shape
        # generate a 1D array only keep the not -1 values
        # for the mcu it switch row first so the result comes columu by column
        self.lookup_table_array1d_row_first = []
        for c in range(lookup_table_cols):
            for r in range(lookup_table_rows):
                if self.lookup_table_array[r, c] == -1:
                    continue
                self.lookup_table_array1d_row_first.append(self.lookup_table_array[r, c])
        # not use it right now
        self.lookup_table_array1d_col_first = []
        for r in range(lookup_table_rows):
            for c in range(lookup_table_cols):
                if self.lookup_table_array[r, c] == -1:
                    continue
                self.lookup_table_array1d_col_first.append(self.lookup_table_array[r, c])

        self.lookup_table_mask = (self.lookup_table_array != -1)
        self.pattern_rows, self.pattern_cols = self.image_pattern_array.shape
        self.data_pos_mapping = []
        for row in range(self.pattern_rows):
            for col in range(self.pattern_cols):
                if self.image_pattern_array[row, col] == 1:
                    self.data_pos_mapping.append((row, col))


    @staticmethod   
    def read_csv_file_to_df(file_path :str) -> pd.DataFrame:
        """
        Reads a CSV file and returns a DataFrame.
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
        

    def convert_adc253_to_plot(self, adc_value :np.ndarray, first: str = "row") -> np.ndarray:
        if isinstance(adc_value, np.ndarray) and np.issubdtype(adc_value.dtype, np.integer):
            adc_value = adc_value.astype(np.float32)
        
        image_plot = np.zeros((self.pattern_rows, self.pattern_cols), dtype=np.float32)
        
        if len(adc_value) != 253:
            raise ValueError(f"Expected a 1D array with 253 elements, got {len(adc_value)} elements")
        
        if first == "row":
            for i in range(len(adc_value)):
                electrode_index = self.lookup_table_array1d_row_first[i]
                # print(f"i: {i}, electrode_index: {electrode_index}")
                row, col = self.data_pos_mapping[electrode_index]
                # print(f"row: {row}, col: {col}")
                image_plot[row, col] = adc_value[i]

        elif first == "col":
            for i in range(len(adc_value)):
                electrode_index = self.lookup_table_array1d_col_first[i]
                row, col = self.data_pos_mapping[electrode_index]
                image_plot[row, col] = adc_value[i]

        else:
            raise ValueError(f"Invalid first value: {first}. Expected 'row' or 'col'.")
        return image_plot


    def convert_origin_adc253_to_plot(self, adc_value: np.ndarray) -> np.ndarray:
        """Convert a 1D adc array (length=253) into the 2D plot matrix.

        This matches the MCU packing logic where each sample is written directly to
        `packet.data[idx*2:idx*2+2]`, i.e. the 1D array is already indexed by
        `electrode_index` in range [0..252].

        Parameters
        ----------
        adc_value : np.ndarray
            1D array with 253 elements, where adc_value[electrode_index] is the
            value for that electrode.
        """
        if isinstance(adc_value, np.ndarray) and np.issubdtype(adc_value.dtype, np.integer):
            adc_value = adc_value.astype(np.float32)

        if len(adc_value) != 253:
            raise ValueError(f"Expected a 1D array with 253 elements, got {len(adc_value)} elements")

        image_plot = np.zeros((self.pattern_rows, self.pattern_cols), dtype=np.float32)

        for electrode_index in range(253):
            row, col = self.data_pos_mapping[electrode_index]
            image_plot[row, col] = adc_value[electrode_index]

        return image_plot


    def convert_adc3210_to_plot(self, adc_value :np.ndarray) -> np.ndarray:
        if isinstance(adc_value, np.ndarray) and np.issubdtype(adc_value.dtype, np.integer):
            adc_value = adc_value.astype(np.float32)

        # check the shape of the adc_value
        if adc_value.shape != (32, 10):
            raise ValueError(f"Expected a 2D array with shape (32, 10), got {adc_value.shape}")
        
        processed_adc = adc_value.copy()
        processed_adc[~self.lookup_table_mask] = np.inf
        image_plot = np.zeros((self.pattern_rows, self.pattern_cols), dtype=np.float32)
        # mapping the adc_value to the image pattern
        adc_shape_rows, adc_shape_cols = processed_adc.shape
        for row in range(adc_shape_rows):
            for col in range(adc_shape_cols):
                if self.lookup_table_mask[row, col] == -1:
                    continue
                electrode_index = self.lookup_table_array[row, col]
                data_row, data_col = self.data_pos_mapping[electrode_index]
                # if electrode_index == 5:
                #     print(f"row: {row}, col: {col}")
                #     print(self.data_pos_mapping[electrode_index])

                image_plot[data_row, data_col] = adc_value[row, col]

        return image_plot
    
    def convert_plot_back_to_adc3210(self, plot_matrix :np.ndarray, enable_chck: bool = True) -> np.ndarray:
        adc_shape_rows, adc_shape_cols = self.lookup_table_array.shape
        adc_matrix = np.zeros((adc_shape_rows, adc_shape_cols), dtype=np.float32)
        
        for electrode_index, (data_row, data_col) in enumerate(self.data_pos_mapping):
            plot_value = plot_matrix[data_row, data_col]
            
            positions = np.where(self.lookup_table_array == electrode_index)
            
            for row, col in zip(positions[0], positions[1]):
                adc_matrix[row, col] = plot_value
        
        if enable_chck:
            # -1 positions are not used so set them to inf
            adc_matrix[~self.lookup_table_mask] = np.inf
            # if the value is equal to 0, set it to 0.5
            adc_matrix[adc_matrix == 0] = 0.5

        return adc_matrix

    def plot_mapped_matrix(self, plot_matrix :np.ndarray) -> None:
        plt.figure(figsize=(6, 12))
        plt.imshow(plot_matrix, cmap='viridis')
        plt.colorbar(label='Resistance (Ω)')
        plt.title('Mapped Rsensor Heatmap')
        plt.xlabel('Channel Index')
        plt.ylabel('Channel Index')
        plt.show()
        
    def get_image_pattern_shape(self):
        """
        Returns the shape of the image pattern.
        """

        return self.image_pattern_array.shape


# test the class
if __name__ == "__main__":
    converter = DataFormatConverter()
    
    # # Test 32x10 ADC data conversion
    # adc_value = np.random.randint(0, 4096, size=(32, 10))
    # print(f"ADC Value: {adc_value}")
    # plot_matrix = converter.convert_adc3210_to_plot(adc_value)
    # converter.plot_mapped_matrix(plot_matrix)

    # # Test 1D ADC253 data conversion 253
    # adc_value = np.random.randint(0, 4096, size=(253,))
    # print(f"ADC Value: {adc_value}")
    # plot_matrix = converter.coonvert_adc253_to_plot(adc_value, first="row")
    # converter.plot_mapped_matrix(plot_matrix)

    # read the CSV file
    # df = DataFormatConverter.read_csv_file_to_df(file_path)
    # # print 6 row and 6 column data
    # print(df.iloc[6, 6])
    # if df is not None:
    #     adc_value_converted = converter.convert_plot_back_to_adc3210(df.values)
    #     print(f"Converted ADC Value: {adc_value_converted}")
    #     # adc_value_converted
    #     adc_value_converted_check = pd.DataFrame(adc_value_converted)
    #     print(adc_value_converted_check.iloc[5,4])
    #     print(adc_value_converted_check)

    # adc_value_converted[0,7] = 50000
    # adc_value_converted[1,7] = 50000
    # adc_value_converted[2,7] = 50000

    # plot_matrix = converter.convert_adc3210_to_plot(adc_value_converted)
    # converter.plot_mapped_matrix(plot_matrix)
    # print(plot_matrix)



    # test= DataFormatConverter()

    # print(test.get_image_pattern_shape())