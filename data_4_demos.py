from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

import json
import vertexai
import requests
import time



class Data4Demos:
    MODEL_NAME = "gemini-2.5-flash-preview-05-20"

    def __init__(self, project_id, location, n_tables, industry, n_rows_per_table, dataset_name=None):
        self.project_id = project_id
        self.location = location
        self.n_tables = n_tables
        self.industry = industry
        self.n_rows_per_table = n_rows_per_table
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.MODEL_NAME)
        self.schema_json = self.generate_schema_json()
        self.dataset_name = dataset_name if dataset_name else "national_grid_data"

    def generate_schema_json(self):
        base_prompt_schema = f"""You are an expert data modeler.
                Your task is to generate the schema description for {self.n_tables}
                Table names should relate to the {self.industry} industry.
                Data is about ESG reporting and should be relevant to the sustainability of the company. 
                Generate both dimension and fact tables.
                """

        schema_json_format = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column_name": {"type": "string"},
                                "column_type": {
                                    "type": "string",
                                    "enum": [
                                        "STRING",
                                        "INT64",
                                        "FLOAT64",
                                        "BOOL",
                                        "TIMESTAMP",
                                        "DATE",
                                        "ARRAY",
                                        "STRUCT",
                                    ],
                                },
                            },
                            "required": ["column_name", "column_type"],
                        },
                    },
                },
                "required": ["table_name", "columns"],
            },
        }
        response = self.model.generate_content(
            base_prompt_schema,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema_json_format,
            ),
        )
        return json.loads(response.text)

    def generate_schema_if_not_exists(self, dataset_name):
        client = bigquery.Client(project=self.project_id)
        dataset_ref = client.dataset(dataset_name)

        try:
            client.get_dataset(dataset_ref)
            print(f"Dataset {dataset_name} already exists.")
        except NotFound:
            print(f"Dataset {dataset_name} not found. Creating dataset.")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = self.location
            client.create_dataset(dataset)
            print(f"Dataset {dataset_name} created.")

    def create_tables(self, dataset_name):
        client = bigquery.Client(project=self.project_id)
        ddl_statements = self.generate_bigquery_ddl(dataset_name, self.schema_json)
        for ddl in ddl_statements:
            client.query(ddl).result()
            print(f"Executed DDL: {ddl}")

    def generate_bigquery_ddl(self, dataset_name, json_schema):
        ddl_statements = []
        for table in json_schema:
            table_name = table["table_name"]
            columns = table["columns"]
            ddl = f"CREATE OR REPLACE TABLE `{dataset_name}.{table_name}` (\n"
            for column in columns:
                column_name = column["column_name"]
                column_type = column["column_type"]
                ddl += f"    `{column_name}` {column_type},\n"
            ddl = ddl.rstrip(",\n") + "\n);"
            ddl_statements.append(ddl)
        return ddl_statements

    def generate_data_for_all_tables(self):
        for table in self.schema_json:
            print(table["columns"])
            data = self.generate_data_for_table(table["table_name"], table["columns"], self.n_rows_per_table)
            self.load_data_into_bigquery_table(table["table_name"], data)

    def load_data_into_bigquery_table(self, table_name, data):
        client = bigquery.Client(project=self.project_id)
        
        # First, ensure the table exists and get a fresh reference
        dataset_ref = client.dataset(self.dataset_name)
        table_ref_full = f"{self.project_id}.{self.dataset_name}.{table_name}"
        
        try:
            # Try to get the table to check if it exists
            try:
                table = client.get_table(table_ref_full)
                print(f"Table {table_name} exists, proceeding with data insertion.")
            except Exception as e:
                print(f"Error checking table {table_name}: {e}")
                print(f"Attempting to recreate table {table_name}...")
                
                # Find the table schema from our schema_json
                table_schema = None
                for t in self.schema_json:
                    if t["table_name"] == table_name:
                        table_schema = t["columns"]
                        break
                
                if table_schema:
                    # Recreate the table
                    ddl = self.generate_bigquery_ddl(self.dataset_name, [{"table_name": table_name, "columns": table_schema}])[0]
                    client.query(ddl).result()
                    print(f"Recreated table with DDL: {ddl}")
                else:
                    print(f"Could not find schema for table {table_name}")
                    return
            
            # Get a fresh reference to the table
            table_ref = client.dataset(self.dataset_name).table(table_name)
            
            # Insert the data with retry logic
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    errors = client.insert_rows_json(table_ref, data)
                    if errors:
                        print(f"Errors occurred while inserting data into {table_name}: {errors}")
                        retry_count += 1
                    else:
                        print(f"Data successfully loaded into {table_name}")
                        success = True
                except Exception as insert_error:
                    print(f"Error inserting data (attempt {retry_count+1}): {insert_error}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying in 2 seconds...")
                        time.sleep(2)
            
            if not success:
                print(f"Failed to insert data after {max_retries} attempts.")
                
        except Exception as outer_error:
            print(f"Unexpected error working with table {table_name}: {outer_error}")

    def generate_data_for_table(self, table_name, table_schema, num_rows):
        base_prompt_data = f"""You are an expert data generator.
                Your task is to generate {num_rows} rows of data for the given table {table_name} with the schema {table_schema}.
                Output should be a list of JSONs where each element of the list is a row with the following structure:
                [ {{ column_name_1': 'value_1', ..., 'column_name_n': 'value_N'}},...,{{...}}]
                Always produce the raw data without any comments or text explanations.
                """
        response = self.model.generate_content(
            base_prompt_data
            )
        trimmed_response = (
            response.candidates[0]
            .content.parts[0]
            .text.replace("```json", "")
            .replace("```", "")
        )
        print(trimmed_response)
        
        # Clean up the response to handle potential comment lines or other JSON issues
        # Remove any lines that start with // which are comments in JSON
        cleaned_response = '\n'.join([line for line in trimmed_response.split('\n') if not line.strip().startswith('//')])
        
        try:
            parsed_response = json.loads(cleaned_response)
            print(parsed_response)
            return parsed_response
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print("Attempting to fix the JSON response...")
            
            # Try to fix common JSON issues
            # 1. Fix trailing commas in arrays which are not valid in JSON
            cleaned_response = cleaned_response.replace(",\n]", "\n]").replace(",]", "]")
            
            # 2. Check for missing quotes around property names
            if "Expecting property name enclosed in double quotes" in str(e):
                import re
                # Find property names without quotes and add quotes
                # This regex looks for property names that aren't properly quoted
                pattern = r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)'
                cleaned_response = re.sub(pattern, r'\1"\2"\3', cleaned_response)
            
            try:
                parsed_response = json.loads(cleaned_response)
                print("Fixed JSON successfully!")
                print(parsed_response)
                return parsed_response
            except json.JSONDecodeError as e2:
                print(f"Could not fix JSON: {e2}")
                # Try a more aggressive approach for really problematic JSON
                try:
                    # Sometimes there's a mix of single and double quotes or other issues
                    # Let's try to normalize the JSON structure
                    import re
                    # Replace single quotes with double quotes (careful with nested quotes)
                    # First, escape any existing double quotes inside values
                    cleaned_response = re.sub(r'([^\\])"', r'\1\\"', cleaned_response)
                    # Now replace all single quotes with double quotes
                    cleaned_response = cleaned_response.replace("'", '"')
                    # Fix double escaped quotes
                    cleaned_response = cleaned_response.replace('\\\\"', '\\"')
                    
                    parsed_response = json.loads(cleaned_response)
                    print("Fixed JSON with aggressive approach!")
                    print(parsed_response)
                    return parsed_response
                except json.JSONDecodeError as e3:
                    print(f"Could not fix JSON with aggressive approach: {e3}")
                    # As a fallback, return a minimal valid dataset based on the schema
                    print("Generating fallback data...")
                    fallback_data = []
                    for i in range(min(10, num_rows)):  # Generate at least some data
                        row = {}
                        for column in table_schema:
                            col_name = column["column_name"]
                            col_type = column["column_type"]
                            if col_type == "INT64":
                                row[col_name] = i + 1
                            elif col_type == "FLOAT64":
                                row[col_name] = float(i + 1) * 10.5
                            elif col_type == "BOOL":
                                row[col_name] = (i % 2 == 0)
                            else:  # Default to STRING for other types
                                row[col_name] = f"{col_name}_value_{i+1}"
                        fallback_data.append(row)
                    return fallback_data

    

    def _map_bigquery_to_json_type(self, bigquery_type):
        type_mapping = {
            "STRING": "string",
            "INTEGER": "integer",
            "FLOAT": "number",
            "BOOLEAN": "boolean",
            "TIMESTAMP": "string",
            "DATE": "string",
            "ARRAY": "array",
            "STRUCT": "object",
        }
        return type_mapping.get(bigquery_type, "string")
    

national_grid=Data4Demos(
    project_id="jsk-dataplex-demo-380508", 
    location="us-central1", 
    n_tables=5, 
    industry="""As National Grid in UK, We develop, own and maintain the physical infrastructure, such as the pylons and cables, needed to move the electricity generated from windfarms and power sources around the country.We own and maintain the high-voltage electricity transmission network in England and Wales. Every time a phone is plugged in, or a switch is turned on, we've played a part, connecting you to the electricity you need.  We take electricity generated from windfarms and other power sources and transport it through our network of pylons, overhead lines, cables, and substations. It then goes on to separate lower voltage local distribution networks, which connect directly to homes and businesses. We're investing for the future, connecting more and more low carbon electricity to our network - it's a crucial role and pivotal in turning the UK's net zero ambitions into reality.""", 
    n_rows_per_table=50,
    dataset_name="national_grid_data"
)
national_grid.generate_schema_if_not_exists(national_grid.dataset_name)
national_grid.create_tables(national_grid.dataset_name)
national_grid.generate_data_for_all_tables()

