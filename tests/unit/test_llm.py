"""
Tests for SAIL LLM Integration
"""

import pytest

from src.core.llm import (
    Constitution,
    ConstitutionalPrinciple,
    GenerationConfig,
    GenerationResult,
    IntentExtractor,
    LLMProviderFactory,
    LLMProviderType,
    Message,
    OllamaProvider,
    LlamaCppProvider,
    ParsedResponse,
    PromptLibrary,
    PromptRole,
    PromptTemplate,
    ResponseParser,
    ResponseType,
    UncertaintyDetector,
    get_prompt_library,
)


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(role="assistant", content="Hi there")
        assert msg.to_dict() == {"role": "assistant", "content": "Hi there"}


class TestGenerationConfig:
    """Tests for GenerationConfig class."""

    def test_default_config(self):
        """Test default generation configuration."""
        config = GenerationConfig()
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.stop_sequences == []

    def test_custom_config(self):
        """Test custom generation configuration."""
        config = GenerationConfig(
            max_tokens=1024,
            temperature=0.5,
            stop_sequences=["END"],
        )
        assert config.max_tokens == 1024
        assert config.temperature == 0.5
        assert config.stop_sequences == ["END"]


class TestGenerationResult:
    """Tests for GenerationResult class."""

    def test_result_creation(self):
        """Test creating a generation result."""
        result = GenerationResult(
            content="Hello, world!",
            model="llama3",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        assert result.content == "Hello, world!"
        assert result.model == "llama3"

    def test_result_usage(self):
        """Test getting usage information."""
        result = GenerationResult(
            content="Test",
            model="test",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_result_usage_with_none(self):
        """Test usage with None values."""
        result = GenerationResult(content="Test", model="test")
        assert result.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class TestOllamaProvider:
    """Tests for OllamaProvider class."""

    def test_provider_creation(self):
        """Test creating an Ollama provider."""
        provider = OllamaProvider(model="llama3")
        assert provider.model == "llama3"
        assert provider.provider_type == LLMProviderType.OLLAMA

    def test_provider_custom_url(self):
        """Test creating provider with custom URL."""
        provider = OllamaProvider(
            model="mistral",
            base_url="http://custom:11434",
        )
        assert provider.base_url == "http://custom:11434"


class TestLlamaCppProvider:
    """Tests for LlamaCppProvider class."""

    def test_provider_creation(self):
        """Test creating a llama.cpp provider."""
        provider = LlamaCppProvider(model="local")
        assert provider.model == "local"
        assert provider.provider_type == LLMProviderType.LLAMACPP


class TestLLMProviderFactory:
    """Tests for LLMProviderFactory class."""

    def test_create_ollama_provider(self):
        """Test creating Ollama provider via factory."""
        provider = LLMProviderFactory.create(
            provider_type="ollama",
            model="llama3",
        )
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3"

    def test_create_llamacpp_provider(self):
        """Test creating llama.cpp provider via factory."""
        provider = LLMProviderFactory.create(
            provider_type="llamacpp",
            model="local",
        )
        assert isinstance(provider, LlamaCppProvider)

    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMProviderFactory.create(
                provider_type="unknown",
                model="test",
            )


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_template_creation(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="test",
            template="Hello, $name!",
            role=PromptRole.USER,
        )
        assert template.name == "test"
        assert "name" in template.required_variables

    def test_template_render(self):
        """Test rendering a template."""
        template = PromptTemplate(
            name="greeting",
            template="Hello, $name! You are in $location.",
        )
        result = template.render(name="Alice", location="Boston")
        assert result == "Hello, Alice! You are in Boston."

    def test_template_render_missing_variable(self):
        """Test rendering with missing variable raises error."""
        template = PromptTemplate(
            name="test",
            template="Hello, $name!",
        )
        with pytest.raises(ValueError, match="Missing required variables"):
            template.render()

    def test_template_default_values(self):
        """Test template with default values."""
        template = PromptTemplate(
            name="test",
            template="Hello, $name!",
            default_values={"name": "User"},
        )
        result = template.render()
        assert result == "Hello, User!"

    def test_template_validate(self):
        """Test template validation."""
        template = PromptTemplate(
            name="test",
            template="$a and $b",
        )
        missing = template.validate(a="value")
        assert "b" in missing
        assert "a" not in missing


class TestPromptLibrary:
    """Tests for PromptLibrary class."""

    def test_library_creation(self):
        """Test creating a prompt library."""
        library = PromptLibrary()
        assert len(library.list_templates()) > 0

    def test_library_get_template(self):
        """Test getting a template from library."""
        library = PromptLibrary()
        template = library.get("system_base")
        assert template is not None
        assert template.role == PromptRole.SYSTEM

    def test_library_get_nonexistent(self):
        """Test getting nonexistent template."""
        library = PromptLibrary()
        assert library.get("nonexistent") is None

    def test_library_render(self):
        """Test rendering from library."""
        library = PromptLibrary()
        result = library.render("system_base")
        assert "SAIL" in result
        assert "SOVEREIGNTY" in result

    def test_library_list_by_role(self):
        """Test listing templates by role."""
        library = PromptLibrary()
        system_templates = library.list_templates(role=PromptRole.SYSTEM)
        assert "system_base" in system_templates

    def test_global_library(self):
        """Test global prompt library instance."""
        library1 = get_prompt_library()
        library2 = get_prompt_library()
        assert library1 is library2


class TestConstitution:
    """Tests for Constitution class."""

    def test_constitution_creation(self):
        """Test creating a constitution."""
        constitution = Constitution()
        assert len(constitution._checks) > 0

    def test_no_external_sharing_violation(self):
        """Test detection of external data sharing."""
        constitution = Constitution()
        response = "You should send your data to our cloud server for analysis."
        violations = constitution.validate(response)
        assert any(v.principle == ConstitutionalPrinciple.SOVEREIGNTY for v in violations)

    def test_no_shaming_violation(self):
        """Test detection of shaming language."""
        constitution = Constitution()
        response = "You should have known better. How could you not know this?"
        violations = constitution.validate(response)
        assert any(v.principle == ConstitutionalPrinciple.NON_JUDGMENT for v in violations)

    def test_compliant_response(self):
        """Test a compliant response passes validation."""
        constitution = Constitution()
        response = "Here's what you can do in this situation. Consider consulting a lawyer for specific legal advice."
        assert constitution.is_compliant(response)

    def test_medical_disclaimer_warning(self):
        """Test medical content without disclaimer."""
        constitution = Constitution()
        response = "Take two aspirin for the headache and you'll feel better."
        violations = constitution.validate(response)
        # Should warn about missing medical professional recommendation
        humility_violations = [
            v for v in violations if v.principle == ConstitutionalPrinciple.HUMILITY
        ]
        assert len(humility_violations) > 0

    def test_medical_with_disclaimer_ok(self):
        """Test medical content with proper disclaimer."""
        constitution = Constitution()
        response = "For headaches, rest can help. However, you should consult a doctor if symptoms persist."
        violations = constitution.validate(response)
        # Should not have humility violations for medical disclaimer
        medical_violations = [
            v
            for v in violations
            if v.principle == ConstitutionalPrinciple.HUMILITY
            and "medical" in v.description.lower()
        ]
        assert len(medical_violations) == 0


class TestUncertaintyDetector:
    """Tests for UncertaintyDetector class."""

    def test_certain_response(self):
        """Test detection of certain response."""
        response = "The answer is 42. This is definitely correct."
        uncertainty = UncertaintyDetector.detect_uncertainty(response)
        assert uncertainty < 0.3

    def test_uncertain_response(self):
        """Test detection of uncertain response."""
        response = "I'm not sure, but it might be around 42. This could vary depending on circumstances."
        uncertainty = UncertaintyDetector.detect_uncertainty(response)
        assert uncertainty > 0.3

    def test_highly_uncertain_response(self):
        """Test detection of highly uncertain response."""
        response = "I really don't know. This is beyond my expertise. You should ask a professional."
        uncertainty = UncertaintyDetector.detect_uncertainty(response)
        assert uncertainty > 0.6

    def test_should_recommend_professional(self):
        """Test professional recommendation detection."""
        uncertain = "I don't know, you should really consult an expert about this."
        certain = "The speed limit in residential areas is typically 25 mph."

        assert UncertaintyDetector.should_recommend_professional(uncertain)
        assert not UncertaintyDetector.should_recommend_professional(certain)


class TestResponseParser:
    """Tests for ResponseParser class."""

    def test_parse_plain_text(self):
        """Test parsing plain text response."""
        response = "This is a simple text response."
        parsed = ResponseParser.parse(response)
        assert parsed.type == ResponseType.TEXT
        assert parsed.content == response

    def test_parse_numbered_steps(self):
        """Test parsing numbered steps."""
        response = """Here's what to do:
1. First, stay calm
2. Second, assess the situation
3. Third, take action"""
        parsed = ResponseParser.parse(response)
        assert parsed.type == ResponseType.STEPS
        assert len(parsed.content) == 3

    def test_parse_bullet_list(self):
        """Test parsing bullet list."""
        response = """Things to consider:
- First item
- Second item
- Third item"""
        parsed = ResponseParser.parse(response)
        assert parsed.type == ResponseType.LIST
        assert len(parsed.content) == 3

    def test_extract_json(self):
        """Test extracting JSON from response."""
        response = """Here's the data:
```json
{"name": "test", "value": 42}
```"""
        data = ResponseParser.extract_json(response)
        assert data == {"name": "test", "value": 42}

    def test_clean_response(self):
        """Test cleaning response."""
        response = "Sure, here's what I found:\n\n\n\nThe answer is 42."
        cleaned = ResponseParser.clean_response(response)
        assert "Sure" not in cleaned
        assert "\n\n\n" not in cleaned
        assert "42" in cleaned

    def test_truncate_response(self):
        """Test truncating response."""
        response = "This is a long response. It has multiple sentences. We need to truncate it."
        truncated = ResponseParser.truncate_response(response, 50)
        assert len(truncated) <= 50

    def test_parse_as_guidance(self):
        """Test parsing as guidance format."""
        response = """Here's what you should know about this situation.

1. Stay calm and collected
2. Assess your options
3. Take appropriate action

Warning: This is time-sensitive.

Note: This is not legal advice."""
        parsed = ResponseParser.parse_as_guidance(response)
        assert parsed.type == ResponseType.GUIDANCE
        assert len(parsed.content["steps"]) == 3


class TestIntentExtractor:
    """Tests for IntentExtractor class."""

    def test_extract_traffic_intent(self):
        """Test extracting traffic-related intent."""
        query = "I just got pulled over by a cop. What should I do?"
        intents = IntentExtractor.extract_intent(query)
        assert len(intents) > 0
        assert intents[0][0] == "legal_traffic"

    def test_extract_scam_intent(self):
        """Test extracting scam-related intent."""
        query = "Someone is asking for my bank details. Is this a scam?"
        intents = IntentExtractor.extract_intent(query)
        intent_names = [i[0] for i in intents]
        assert "safety_scam" in intent_names or "financial_transfer" in intent_names

    def test_extract_medical_intent(self):
        """Test extracting medical emergency intent."""
        query = "Someone is hurt and bleeding. What do I do?"
        intents = IntentExtractor.extract_intent(query)
        assert any("emergency" in i[0] for i in intents)

    def test_get_primary_intent(self):
        """Test getting primary intent."""
        query = "My boss fired me today. What are my rights?"
        intent = IntentExtractor.get_primary_intent(query)
        assert intent == "legal_employment"

    def test_get_domain(self):
        """Test getting domain from intent."""
        assert IntentExtractor.get_domain("legal_traffic") == "legal"
        assert IntentExtractor.get_domain("safety_scam") == "safety"
        assert IntentExtractor.get_domain("emergency_medical") == "emergency"
        assert IntentExtractor.get_domain("unknown") is None

    def test_no_intent_detected(self):
        """Test when no intent is detected."""
        query = "What's the weather like today?"
        intent = IntentExtractor.get_primary_intent(query)
        assert intent is None


class TestParsedResponse:
    """Tests for ParsedResponse class."""

    def test_text_as_text(self):
        """Test text response as text."""
        parsed = ParsedResponse(
            type=ResponseType.TEXT,
            content="Hello",
            raw="Hello",
        )
        assert parsed.as_text() == "Hello"

    def test_steps_as_text(self):
        """Test steps response as text."""
        parsed = ParsedResponse(
            type=ResponseType.STEPS,
            content=["First", "Second", "Third"],
            raw="",
        )
        text = parsed.as_text()
        assert "1. First" in text
        assert "2. Second" in text

    def test_list_as_text(self):
        """Test list response as text."""
        parsed = ParsedResponse(
            type=ResponseType.LIST,
            content=["Item A", "Item B"],
            raw="",
        )
        text = parsed.as_text()
        assert "• Item A" in text
        assert "• Item B" in text
