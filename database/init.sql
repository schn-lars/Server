CREATE TABLE IF NOT EXISTS birds (
	id SERIAL PRIMARY KEY,
	trivial_name TEXT,
	species_name TEXT,
	city TEXT,
	canton TEXT,
	x DECIMAL(18, 15),
	y DECIMAL(18,15),
	counter INTEGER,
	year_number INTEGER
);

