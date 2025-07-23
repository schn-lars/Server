CREATE TABLE IF NOT EXISTS adresses (
	id SERIAL PRIMARY KEY,
	street TEXT,
	number TEXT,
	zip INT,
	zip_label TEXT,
	name TEXT,
	canton TEXT,
	coord_x DECIMAL(18,15),
	coord_y DECIMAL(18,15)
);
