toolConfig = {
    "tools": [
        {
            "toolSpec": {
                "name": "doc_retrieval",
                "description": "Retrieve entire document from docDB. To be used only when it's necessary to retrieve all information in a document",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "filter": {
                                "type": "string",
                                "description": "A MongoDB query to pass to the function"
                            }
                        },
                        "required": ["filter"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "projection_retrieval",
                "description": "Retrieve multiple documents from docDB with only specific field information. Used when most of the document is not necessary to answer natural language query",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                                    "filter": {
                                        "type": "string",
                                        "description": "A MongoDB query to pass to the function"
                                    },
                                    "fieldNameList": {
                                        "type": "string",
                                        "description": "A list of field names following JSON format to retrieve from the document. The string shouldn't contain the value, just the key you would need to access the value in a metadata schema document."
                                    }
                        },
                        "required": ["filter", "fieldNameList"]
                    }
                }
            }
        }
    ],
    "toolChoice": {
        "auto":{},
    }
}
