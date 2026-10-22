import pandera as pa

def check(df):
    """
    Verify that after quarantine, total_distance_km has no null values.
    """
    schema = pa.DataFrameSchema({
        'total_distance_km': pa.Column(pa.Float(), nullable=False)
    })
    schema.validate(df)
