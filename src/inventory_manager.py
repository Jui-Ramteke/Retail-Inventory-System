def calculate_inventory_metrics(forecasted_sales, lead_time=3, service_level_factor=1.65):
    """
    Standard Inventory Formula:
    Reorder Point (ROP) = (Avg Daily Sales * Lead Time) + Safety Stock
    Safety Stock = Z * std_dev * sqrt(Lead Time)
    """
    avg_sales = forecasted_sales.mean()
    std_dev = forecasted_sales.std()
    
    safety_stock = service_level_factor * std_dev * (lead_time ** 0.5)
    reorder_point = (avg_sales * lead_time) + safety_stock
    
    return round(safety_stock), round(reorder_point)