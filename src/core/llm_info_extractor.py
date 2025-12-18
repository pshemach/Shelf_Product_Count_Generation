from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import numpy as np
import cv2
import base64
import os
import sys
from typing import Literal, Dict, Any, Optional
from dotenv import load_dotenv
from src.exception import MerchandiserException
from src.logs import logging

load_dotenv()

LLMProvider = Literal["openai", "anthropic"]

class Product(BaseModel):
    """Pydantic model for structured product information output."""
    brand: str = Field(description="Extract the exact brand name from visible labels or logos")
    product_name: str = Field(description="Full product name or title as shown on the packaging")
    variant: str = Field(description="Identify the variant or type")
    primary_colors: list[str] = Field(description="List the dominant colors of the packaging")
    packaging_type: str = Field(description="Type of container (e.g., 'Bottle', 'Tube', 'Jar', 'Box', 'Can')")
    shape: str = Field(description="Describe the container shape (e.g., 'Cylindrical', 'Rectangular', 'Oval')")
    size_volume: str = Field(description="Extract exact size/volume from label (e.g., '200ml', '500g') or estimate if not visible")
    logo_description: str = Field(description="Brief description of the brand logo appearance and position.")
    key_text: list[str] = Field(description="Extract main visible text or claims on the label")
    distinctive_elements: list[str] = Field(description="Any unique design features, patterns, or color combinations that make this product recognizable")

class ProductInfo(BaseModel):
    """Wrapper model for product information."""
    product: Product

class LLMInfoExtractor:
    """
    A class to extract product information from product images using vision-capable LLMs.
    Supports both OpenAI and Anthropic providers.
    """
    def __init__(
        self, 
        llm_provider: LLMProvider = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0
        ):
        
        self.llm_provider = llm_provider
        self.temperature = temperature
        
        if model_name is None:
            if llm_provider == "openai":
                model_name = "gpt-4o-mini"
            elif llm_provider == "anthropic":
                model_name = "claude-sonnet-4-5-20250929"
        
        self.model_name = model_name
        
        if api_key is None:
            if llm_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif llm_provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            logging.error(f"API key not found for {llm_provider}")
            raise MerchandiserException(f"API key not found for {llm_provider}", sys)
        
        self.api_key = api_key
        
        try:
            if llm_provider == "openai":
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=self.temperature
                )
            elif llm_provider == "anthropic":
                self.llm = ChatAnthropic(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=self.temperature
                )
                
            self.parser = JsonOutputParser(pydantic_object=ProductInfo)
            logging.info(f"LLM initialized successfully for {llm_provider.upper()} with model {self.model_name}")
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            raise MerchandiserException(e, sys) from e
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """
        Encode the image to base64 string.
        """
        try:
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image to base64: {e}")
            raise MerchandiserException(e, sys) from e
    def extract(self, product: np.array):
        """
        Extract text from the product image.
        """
        try:   
            base64_image = self._encode_image_to_base64(product)
            
            prompt = """
            Analyze the product image and extract key identifying features.
                    
            {format_instructions}
                    
            Extract: brand, product_name, variant, primary_colors (list), packaging_type, 
            shape, size_volume, logo_description, key_text (list), distinctive_elements (list).
                    
            Mark unclear features as "Unknown".
            """
            format_instructions = self.parser.get_format_instructions()
            prompt = prompt.format(format_instructions=format_instructions)
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]
            )   
            response = self.llm.invoke([message])
            return self.parser.parse(response.content)
        except Exception as e:
            logging.error(f"Error extracting text from product: {e}")
            raise MerchandiserException(e, sys) from e