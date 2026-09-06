-- Agentic BI Platform schema — dependency-ordered DDL

CREATE TABLE customers (
    customer_unique_id TEXT PRIMARY KEY,
    customer_zip_code_prefix TEXT,
    customer_city TEXT,
    customer_state TEXT
);

CREATE TABLE customer_crosswalk (
    customer_id TEXT PRIMARY KEY,
    customer_unique_id TEXT,
    FOREIGN KEY (customer_unique_id) REFERENCES customers(customer_unique_id)
);

CREATE TABLE product_category_name_translation (
    product_category_name TEXT PRIMARY KEY,
    product_category_name_english TEXT
);

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_category_name TEXT,
    product_name_length INTEGER,
    product_description_length INTEGER,
    product_photos_qty INTEGER,
    product_weight_g NUMERIC,
    product_length_cm NUMERIC,
    product_height_cm NUMERIC,
    product_width_cm NUMERIC,
    FOREIGN KEY (product_category_name)
        REFERENCES product_category_name_translation(product_category_name)
);

CREATE TABLE regions (
    region_id INTEGER PRIMARY KEY,
    region_name TEXT
);

CREATE TABLE state_region_map (
    state_code TEXT PRIMARY KEY,
    region_id INTEGER,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

CREATE TABLE employees (
    employee_id TEXT PRIMARY KEY,
    employee_name TEXT,
    role TEXT,
    region_id INTEGER,
    hire_date DATE,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

CREATE TABLE sellers (
    seller_id TEXT PRIMARY KEY,
    seller_zip_code_prefix TEXT,
    seller_city TEXT,
    seller_state TEXT,
    employee_id TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT,
    order_status TEXT,
    order_purchase_timestamp TIMESTAMP,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customer_crosswalk(customer_id)
);

CREATE TABLE order_items (
    order_id TEXT,
    order_item_id INTEGER,
    product_id TEXT,
    seller_id TEXT,
    shipping_limit_date TIMESTAMP,
    price NUMERIC(10,2),
    freight_value NUMERIC(10,2),
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (seller_id) REFERENCES sellers(seller_id)
);

CREATE TABLE order_payments (
    order_id TEXT,
    payment_sequential INTEGER,
    payment_type TEXT,
    payment_installments INTEGER,
    payment_value NUMERIC(10,2),
    PRIMARY KEY (order_id, payment_sequential),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE order_reviews (
    review_id TEXT PRIMARY KEY,
    order_id TEXT,
    review_score SMALLINT,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TIMESTAMP,
    review_answer_timestamp TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE inventory (
    product_id TEXT PRIMARY KEY,
    stock_quantity INTEGER,
    restock_date DATE,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE returns (
    return_id TEXT PRIMARY KEY,
    order_id TEXT,
    order_item_id INTEGER,
    return_reason TEXT,
    return_date DATE,
    refund_amount NUMERIC(10,2),
    FOREIGN KEY (order_id, order_item_id)
        REFERENCES order_items(order_id, order_item_id)
);
